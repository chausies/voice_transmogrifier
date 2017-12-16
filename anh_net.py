from __future__ import division, print_function
import matplotlib.pylab as P
from keras.models import Model
from keras.regularizers import l2
from keras.layers \
    import Input, Dense, Conv1D, Dropout, Activation, \
    TimeDistributed, add, multiply, Lambda
from keras import backend as KB
from keras.losses import mean_squared_error, mean_absolute_error
from soundfile import read, write
from scipy.signal import resample_poly
from datetime import datetime
import os
import errno

NOW = datetime.now().strftime("%B_%d_%Y_at_%I%M%p")


print("Starting...")
seed       = int(10e7*P.rand())
num_filts  = 100     # Number of filters for the conv layers
K          = 3      # Number of Dilation layers
D          = 9      # Dilation fold (number of dilations per dilation layer).
                    # dilate by factor of 2**D
dil_reg    = 1e-4   # regularization for dilated conv layer
reg_out    = 1e-5   # regularization for output relu's
batch_size = 8
samps      = 2000   # Number of valid samps per sequence
stride     = 600    # hop size
iSNR       = 0.01   # inverse signal to noise ratio. basically, volume
                    # of noise added to input
desired_sr = 16000  # signals are resampled to this rate
mu         = 256    # quantization level
C          = 400    # number of chunks
drop_prob  = 0.1    # Dropout probability for pre-output layer
VAL        = False  # Whether to have validation or not

desc = """
Running with anh_2 output and ajay1+ajay2 input and over samps things. Also
shuffling the training data. Also, properly losing culling data from
dilated convs. Also added dropout to skip outputs. Using rmsprop.
"""

TXT = desc + \
"""
seed       = {0}
num_filts  = {1}     # Number of filters for the conv layers
K          = {2}      # Number of Dilation layers
D          = {3}      # Dilation fold (number of dilations per dilation layer).
                    # dilate by factor of 2**D
dil_reg    = {4:.3}   # regularization for dilated conv layer
reg_out    = {5:.3}   # regularization for output relu's
batch_size = {6}
samps      = {7}   # Number of valid samps per sequence
stride     = {8}    # hop size
iSNR       = {9:.3}   # inverse signal to noise ratio. basically, volume
                    # of noise added to input
desired_sr = {10}  # signals are resampled to this rate
mu         = {11}    # quantization level
C          = {12}    # number of chunks
drop_prob  = {13}    # Dropout probability for pre-output layer
VAL        = {14}  # Whether to have validation or not
""".format(
    seed, num_filts, K, D, dil_reg, reg_out, batch_size, samps,
    stride, iSNR, desired_sr, mu, C, drop_prob, VAL
    )

print(TXT)
P.seed(seed)

# Getting the data
x1, sr = read("new/ajay1.wav")
y1, sr = read("new/anh_synced_ajay1.wav")
y1 = y1[:len(x1)]
x1 = resample_poly(x1, desired_sr, sr)
vx1 = P.sqrt((x1**2).mean())
x1 = x1/vx1*.05
y1 = resample_poly(y1, desired_sr, sr)
vy1 = P.sqrt((y1**2).mean())
y1 = y1/vy1*.05
sr = desired_sr

x2, sr = read("new/ajay2.wav")
y2, sr = read("new/anh_synced_ajay2.wav")
y2 = y2[:len(x2)]
x2 = resample_poly(x2, desired_sr, sr)
vx2 = P.sqrt((x2**2).mean())
x2 = x2/vx2*.05
y2 = resample_poly(y2, desired_sr, sr)
vy2 = P.sqrt((y2**2).mean())
y2 = y2/vy2*.05
sr = desired_sr

x = P.concatenate([x1, x2])
y = P.concatenate([y1, y2])


vx = P.sqrt((x**2).mean())
vy = P.sqrt((y**2).mean())

XMAX = abs(x).max()
YMAX = abs(y).max()
print("Length of signals")
print(len(x))
print(len(y))

receptive_field = K * (2**D * 2) - (K - 1)
receptive_field_ms = receptive_field*1000/sr
print( "Receptive field: {0}".format(receptive_field) )
print( "Receptive field: {0:.4} ms".format(receptive_field_ms) )

# x = P.hstack([x + P.randn(len(x))*.02*v, x + P.randn(len(x))*.01*v, x])
# y = P.hstack([y, y, y])

print("Calculating error stats...")
ah = 8
a = y.reshape(-1)[:-ah]
ax = x.reshape(-1)
print("VARIANCE")
print(P.var(a))
print("STD")
print(P.std(a))
A = P.vstack([a[i:(i-ah)] for i in range(ah)])
AX = P.vstack([ax[i:(i-2*ah)] for i in range(2*ah)])
A = P.vstack([A, AX, ax[2*ah:]])
b = a[ah:]
A = A - P.mean(A, 1).reshape(-1, 1)
b = b - P.mean(b)
print("LMMSE with {0} taps".format(ah))
LMMSE = P.mean((b - A.T.dot(P.inv(P.dot(A, A.T)).dot(A.dot(b))))**2)
print(LMMSE)
print("LMRMSE with {0} taps".format(ah))
print(P.sqrt(LMMSE))

def mu_law(a, mu=256, MAX=None):
  mu = mu-1
  a = P.array(a)
  MAX = a.max() if MAX is None else MAX
  a = a/MAX
  y = (1 + P.sign(a)*P.log(1 + mu*abs(a))/P.log(1 + mu))/2
  inds = P.around(y*mu).astype(P.uint8)
  return inds
def inv_mu_law(y, mu=256, MAX=1.0):
  mu = mu - 1
  y = (2.*y)/mu - 1
  a = P.sign(y)*((1+mu)**abs(y) - 1)/mu
  a = MAX*a
  return a
def KB_inv_mu_law(y, mu=256, MAX=1.0):
  mu = mu - 1
  y = (2.0*y)/mu - 1
  a = KB.sign(y)*(KB.pow(1.+mu, KB.abs(y)) - 1)/mu
  a = MAX*a
  return a

seq_length = receptive_field
factor = 10 if VAL else 1
T = len(x)
print( "Number of samples: {0}".format(T)                )
T_even = (
    (T - (2*seq_length + 1 + samps))//stride + 1
    ) //(batch_size*C*factor) * batch_size*C*factor
print( "Number of frames per epoch: {0}".format(T_even)  )
T_even = stride*(T_even - 1) + (2*seq_length + 1 + samps)
OFFSET = T - T_even     - 1
T = T_even
print( "Number of even samples: {0}".format(T)           )
print( "Number of training samples: {0}".format(T*9//10) )
print( "Number of val samples: {0}".format(T//10)        )

def get_slicer(i, e):
  def slicer(x):
    return x[:, i:e, :]
  return slicer

samp_slicer = Lambda(get_slicer(-samps, None), output_shape=(samps, num_filts))

def dil_layer(original_y, original_x, k, d):
  y = original_y
  x = original_x
  out_y_len = int(original_y.shape[1]) - 2**d
  out_x_len = int(original_x.shape[1]) - 2**(d+1)
  y_slicer = Lambda(get_slicer(2**d, None), output_shape=(out_y_len, num_filts))
  sliced_x = Lambda(get_slicer(2**d, -2**d), 
      output_shape=(out_x_len, num_filts))(original_x)
  sliced_y = y_slicer(original_y)
  slicer = Lambda(get_slicer(1, 1+out_y_len), output_shape=(out_y_len, num_filts))
  tconvyy = Conv1D(num_filts, 2, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="causal", use_bias=True,
      name="dil_layer_k{0}d{1}_tanhy_y".format(k,d))(y)
  tconvyy = y_slicer(tconvyy)
  tconvyx = Conv1D(num_filts, 3, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="valid", use_bias=True,
      name="dil_layer_k{0}d{1}_tanhy_x".format(k,d))(x)
  tconvyx = slicer(tconvyx)
  tconvy = add( [tconvyy, tconvyx] )
  tanhy = Activation('tanh',
      name="dil_layer_k{0}d{1}_tanhy".format(k,d))( tconvy )
  sconvyy = Conv1D(num_filts, 2, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="causal", use_bias=True,
      name="dil_layer_k{0}d{1}_sigmy_y".format(k,d))(y)
  sconvyy = y_slicer(sconvyy)
  sconvyx = Conv1D(num_filts, 3, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="valid", use_bias=True,
      name="dil_layer_k{0}d{1}_sigmy_x".format(k,d))(x)
  sconvyx = slicer(sconvyx)
  sconvy = add( [sconvyy, sconvyx] )
  sigmy = Activation('sigmoid',
      name="dil_layer_k{0}d{1}_sigmy".format(k,d))( sconvy )
  gy = multiply([tanhy, sigmy])
  res_y = Conv1D(num_filts, 1, padding="same", use_bias=True, 
      kernel_regularizer=l2(dil_reg),
      name="dil_layer_k{0}d{1}_pre_res_y".format(k,d))(gy)
  res_y = add([sliced_y, res_y])
  gy = samp_slicer(gy)
  skip_y = Conv1D(num_filts, 1, padding="same",
      kernel_regularizer=l2(dil_reg),
      name="dil_layer_k{0}d{1}_skip_y".format(k,d))(gy)
  tanhx = Conv1D(num_filts, 3, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="valid", use_bias=True, activation='tanh', 
      name="dil_layer_k{0}d{1}_tanhx".format(k,d))(x)
  sigmx = Conv1D(num_filts, 3, dilation_rate=2**d, kernel_regularizer=l2(dil_reg),
      padding="valid", use_bias=True, activation='sigmoid', 
      name="dil_layer_k{0}d{1}_sigmx".format(k,d))(x)
  gx = multiply([tanhx, sigmx])
  res_x = Conv1D(num_filts, 1, padding="same", use_bias=True, 
      kernel_regularizer=l2(dil_reg),
      name="dil_layer_k{0}d{1}_pre_res_x".format(k,d))(gx)
  res_x = add([sliced_x, res_x])
  gx = Lambda(get_slicer(out_y_len-samps, out_y_len), 
      output_shape=(samps, num_filts))(gx)
  skip_x = Conv1D(num_filts, 1, padding="same",
      kernel_regularizer=l2(dil_reg),
      name="dil_layer_k{0}d{1}_skip_x".format(k,d))(gx)
  skip = add([skip_y, skip_x])
  skip = Dropout(drop_prob)(skip)
  return res_y, res_x, skip

y_input = Input(shape=(seq_length + samps, mu), name="y_history_input")
x_input = Input(shape=(2*seq_length+1 + samps, mu), name="x_input")

y_out = y_input
x_out = x_input
y_out = Conv1D(num_filts, 2, dilation_rate=1, padding="causal", 
    use_bias=True, name="initial_causal_y_conv")(y_out)
y_out = Lambda(get_slicer(1, None), 
    output_shape=(seq_length+samps-1, num_filts))(y_out)
x_out = Conv1D(num_filts, 3, dilation_rate=1, padding="valid", 
    use_bias=True, name="initial_x_conv")(x_out)
skip_outs = [None]*K*(D+1)
for k in range(K):
  for d in range(D+1):
    y_out, x_out, skip_out = dil_layer(y_out, x_out, k, d)
    skip_outs[k*(D+1) + d] = skip_out
# slicer = Lambda(slice, output_shape=(seq_length, num_filts))
# skip_outs = [y_out, slicer(x_out)]
out = add(skip_outs)
out = Activation('relu', name="post_dil_conv_relu")(out)
out = Dropout(drop_prob)(out)
out = Conv1D(mu, 1, padding="same", name="pre_output_conv", 
    use_bias=True, kernel_regularizer=l2(reg_out))(out)
out = Activation('relu', name="pre_output_relu")(out)
out = Conv1D(mu, 1, padding="same", use_bias=True,
    name="output_layer")(out)
out = Activation('softmax', name="output_softmax")(out)

model = Model(inputs=[y_input, x_input], outputs=out)

# model.load_weights("December_02_2017_at_1007PM/ep_0006_chunk_0000.h5")
#########################################
# Q = inv_mu_law(P.r_[:mu], mu, YMAX)
# Qk = KB.reshape(Q, (1, 1, -1))
def inv_mu_mse(y_true, y_pred):
  true_inds = KB.cast( KB.argmax(y_true), 'float32')
  pred_inds = KB.cast( KB.argmax(y_pred), 'float32')
  y_true = KB_inv_mu_law(true_inds, mu, YMAX)
  y_pred = KB_inv_mu_law(pred_inds, mu, YMAX)
  return mean_squared_error(y_true, y_pred)
  # y_true = KB.expand_dims(KB.cast(KB_inv_mu_law(true_inds, mu, YMAX), 'float64'), -1)
  # return KB.sum(
  #   KB.square(KB.cast(Qk - y_true, 'float32'))*y_pred,
  #   axis=-1)

def inv_mu_mae(y_true, y_pred):
  true_inds = KB.cast( KB.argmax(y_true), 'float32')
  pred_inds = KB.cast( KB.argmax(y_pred), 'float32')
  y_true = KB_inv_mu_law(true_inds, mu, YMAX)
  y_pred = KB_inv_mu_law(pred_inds, mu, YMAX)
  return mean_absolute_error(y_true, y_pred)
  # y_true = KB.expand_dims(KB.cast(KB_inv_mu_law(true_inds, mu, YMAX), 'float64'), -1)
  # return KB.sum(
  #   KB.abs(KB.cast(Qk - y_true, 'float32'))*y_pred,
  #   axis=-1)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'categorical_crossentropy', inv_mu_mse, inv_mu_mae])


print( "Number of model params: {0}".format( model.count_params() ) )

# raw_input("Press enter to start...")

# # start halfway, where it's exciting
# print("Starting training half-way through signal...")

dirname = NOW + "/"
if not os.path.exists(os.path.dirname(dirname)):
    try:
        os.makedirs(os.path.dirname(dirname))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(dirname + "params.txt", "w") as text_file:
  text_file.write(TXT)

m = (
    (T - (2*seq_length + 1 + samps))//stride + 1
    ) // C
for ep in range(300):
  print("On epoch {0}".format(ep))
  offset = int(P.rand()*OFFSET)
  arm = P.arange(m).reshape(-1, 1)
  arsy_i = P.arange(seq_length + samps).reshape(1, -1)
  arsy_o = P.arange(samps).reshape(1, -1)
  arsx = P.arange(2*seq_length+1 + samps).reshape(1, -1)
  PERM = P.permutation(C*m)*stride + offset
  for c in range(C):
    # if ep==0 and c<C//2:
    #   continue
    print("On chunk {0}/{1}".format(c+1, C))
    # I = c*m*stride + offset
    x_c = P.vstack([ x[PERM[c*m+fn]:(PERM[c*m+fn]+(2*seq_length+1+samps))] for fn in range(m) ])
    x_c = x_c + P.randn(*x_c.shape)*vx*iSNR # add light noise
    x_c[x_c>XMAX] = XMAX
    x_c[x_c<-XMAX] = -XMAX
    x_inds = mu_law(x_c, mu, XMAX)
    x_c = P.zeros((m, 2*seq_length+1+samps, mu))
    x_c[arm, arsx, x_inds] = True
    y_c_i = P.vstack([ y[PERM[c*m+fn]:(PERM[c*m+fn]+seq_length+samps)] for fn in range(m) ])
    y_c_i = y_c_i + P.randn(*y_c_i.shape)*vy*iSNR # add light noise
    y_c_i[y_c_i>YMAX] = YMAX
    y_c_i[y_c_i<-YMAX] = -YMAX
    y_inds = mu_law(y_c_i, mu, YMAX)
    y_c_i = P.zeros((m, seq_length+samps, mu))
    y_c_i[arm, arsy_i, y_inds] = True
    # I = c*m*stride + 1 + offset
    # y_c_o = P.vstack([ y[(I+stride*fn):(I+stride*fn+seq_length)] for fn in range(m) ])
    y_c_o = P.vstack([ y[(PERM[c*m+fn]+1):(PERM[c*m+fn]+1+seq_length+samps)][-samps:] for fn in range(m) ])
    y_c_o[y_c_o>YMAX] = YMAX
    y_c_o[y_c_o<-YMAX] = -YMAX
    y_inds = mu_law(y_c_o, mu, YMAX)
    y_c_o = P.zeros((m, samps, mu))
    y_c_o[arm, arsy_o, y_inds] = True
    # x_train = x_c[:-(m//10), :, :]
    # y_train = y_c[:-(m//10), :, :]
    #
    # x_val = x_c[-(m//10):, :, :]
    # y_val = y_c[-(m//10):, :, :]
    print("starting fit...")
    model.fit([y_c_i, x_c], y_c_o,
              batch_size=batch_size, epochs=1, shuffle=False)#, validation_split=0.1)
    if (c*2) % C == 0:
      model.save_weights(
          dirname + "ep_{0:04}_chunk_{1:04}.h5".format(ep, c)
          )
