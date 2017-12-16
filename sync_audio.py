from __future__ import division
import matplotlib.pylab as P
import librosa
import soundfile as sf
import librosa.display
from scipy.interpolate import PchipInterpolator as mono_interp
from scipy.signal import decimate



def variable_phase_vocoder(D, times_steps, hop_length=None):
    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    # time_steps = P.arange(0, D.shape[1], rate, dtype=P.double)
    # time_steps = P.concatenate([
    #   P.arange(0, D.shape[1]/2, .5, dtype=P.double),
    #   P.arange(D.shape[1]/2, D.shape[1], 2, dtype=P.double)
    #   ])

    # Create an empty output array
    d_stretch = P.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = P.linspace(0, P.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = P.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = P.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):
        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = P.mod(step, 1.0)
        mag = ((1.0 - alpha) * abs(columns[:, 0])
               + alpha * abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * P.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (P.angle(columns[:, 1])
                  - P.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * P.pi * P.around(dphase / (2.0 * P.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch

def close_points(X, s=1):
  lambda_s = 1
  lambda_c = s
  X = P.array(X)
  K, N = X.shape
  M = P.zeros((N, N))
  M[range(N), range(N)] = 2*lambda_c/(N-1) + lambda_s/N
  # M[0,0] -= lambda_c/(N-1)
  # M[-1,-1] -= lambda_c/(N-1)
  d = P.diag(P.ones(N-1), 1)
  M = M - lambda_c*(d + d.T)/(N-1)
  M[0, 0] = lambda_s/N
  M[-1, -1] = lambda_s/N
  M[0, 1] = 0
  M[-1, -2] = 0
  Mi = P.pinv(M)
  smooth_X = (lambda_s/N)*Mi.dot(X.T).T
  return smooth_X

print "Loading Audio..."
# y1, fs = librosa.load("r1.wav")
# y2, fs = librosa.load("r2.wav")
# y1, fs = librosa.load("test_anh.wav")
# y2, fs = librosa.load("test_ajay.wav")
y1, fs = sf.read("anh_2.wav")
y2, fs = sf.read("ajay2.wav")
# Add some simple padding
i1 = P.argmax( y1 > P.sqrt((y1**2).mean())/3 )
i2 = P.argmax( y2 > P.sqrt((y2**2).mean())/3 )
I = max(i1, i2)*2
z1 = y1[i1//5:(i1//5)*2]
y1 = P.hstack([z1]*((I-i1)//len(z1)) + [z1[:((I - i1)%len(z1))]] + [y1])
z2 = y2[i2//5:(i2//5)*2]
y2 = P.hstack([z2]*((I-i2)//len(z2)) + [z2[:((I - i2)%len(z2))]] + [y2])
# y1 = P.concatenate([P.zeros(I - i1), y1])
# y2 = P.concatenate([P.zeros(I - i2), y2])
print("Setting padding to {0:.2f} s".format(I/fs))
# manually downsample by factor of 2
fs = fs//2
y1 = decimate(y1, 2, zero_phase=True)
y2 = decimate(y2, 2, zero_phase=True)
# Normalize loudness
v1 = P.sqrt((y1**2).mean())
v2 = P.sqrt((y2**2).mean())
y1 = y1/v1*.03
y2 = y2/v2*.03
print "Audio lengths (s)"
print len(y1)/fs
print len(y2)/fs

# d = fs//6
# ti = P.argmax(abs(y1)>abs(y1).max()/5)
# tf = P.argmax(abs(y1[::-1])>abs(y1[::-1]).max()/5) - 1
# y1 = y1[ti-d:-(tf-d)]
# ti = P.argmax(abs(y2)>abs(y2).max()/5)
# tf = P.argmax(abs(y2[::-1])>abs(y2[::-1]).max()/5) - 1
# d = int(len(y1) - len(y2) + ti + tf)
# di, df = d//2, (d+1)//2
# y2 = y2[ti-di:-(tf-df)]

n_fft = 4410
hop_size = 2205

print "Starting DTW..."
y1_mfcc = librosa.feature.mfcc(y=y1, sr=fs, 
                              hop_length=hop_size, n_mfcc=80)
y2_mfcc = librosa.feature.mfcc(y=y2, sr=fs, 
                              hop_length=hop_size, n_mfcc=80)
D, wp = librosa.core.dtw(X=y1_mfcc, Y=y2_mfcc, metric='cosine')
print "Doing interpolation and warping..."
wp = wp[::-1, :]
y1_st, y1_end = wp[0, 0]*hop_size, wp[-1, 0]*hop_size
y2_st, y2_end = wp[0, 1]*hop_size, wp[-1, 1]*hop_size
y1 = y1[y1_st:y1_end]
y2 = y2[y2_st:y2_end]
wp[:, 0] = wp[:, 0] - wp[0,0]
wp[:, 1] = wp[:, 1] - wp[0,1]
wp_s = P.asarray(wp) * hop_size / fs
i, I = P.argsort(wp_s[-1, :])
x, y = close_points(
  P.array([wp_s[:,i]/wp_s[-1,i], wp_s[:,I]/wp_s[-1,I]]), s=1)
f = mono_interp(x, y, extrapolate=True)
yc,yo = (y1,y2) if i==1 else (y2, y1)
l_hop = 64
stft = librosa.stft(yc, n_fft=512, hop_length=l_hop)
z = len(yo)//l_hop + 1
t = P.arange(0, 1, 1/z)
time_steps = P.clip( f(t) * stft.shape[1], 0, None )
print "Beginning vocoder warping..."
warped_stft = variable_phase_vocoder(stft, time_steps, hop_length=l_hop)
y_warp = librosa.istft(warped_stft, hop_length=l_hop)
# print "Writing warped signal to file..."
# librosa.output.write_wav("warped.wav", y_warp, fs, norm=True)
# P.plot(wp_s[:,1]/wp_s[-1,1], wp_s[:,0]/wp_s[-1,1], 'o')
# x, y = close_points(
#   P.array([wp_s[:,1]/wp_s[-1,1], wp_s[:,0]/wp_s[-1,1]]), s=1)
# f = mono_interp(x, y, extrapolate=True)
# t = P.linspace(0, 1, 500)
# P.plot(t, f(t))
# # f_smooth = US(wp_s[:,1]/wp_s[-1,1], wp_s[:,0]/wp_s[-1,1], s=.008)
# # P.plot(t, f_smooth(t))
# # P.plot(x, y, 'x')
# P.show()
#
# # P.plot(t, wp_s[:, 0])
# # P.plot(t, wp_s[:, 1])
#

# print "Plotting DTW result..."
# fig = P.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# librosa.display.specshow(D, x_axis='time', y_axis='time',
#                           cmap='gray_r', hop_length=hop_size)
# imax = ax.imshow(D, cmap=P.get_cmap('gray_r'),
#                 origin='lower', interpolation='nearest', aspect='auto')
# ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
# P.title('Warping Path on Acc. Cost Matrix $D$')
# P.colorbar()
# P.show()

#
l = min(len(y_warp), len(y2))
y = y_warp[:l] + y2[:l]#*.8 # because my male voice overpower's Anh's female voice

print "Writing syned reading to file..."
sf.write("y1_warp.wav", y_warp[:l], fs)
sf.write("y2.wav", y2[:l], fs)
sf.write("synced.wav", y, fs)
