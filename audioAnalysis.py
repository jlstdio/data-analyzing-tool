import librosa
import numpy as np
from matplotlib import pyplot as plt
from featureExtractor import *
np.set_printoptions(precision=32, suppress=True)

# path ###################
path = "../data/splitData/audio/audio0/audio0_300.wav"
# y, sr = sf.read('data/splitData/acc0/acc0_1.wav')
# y, sr = sf.read('data/0/selected/audio0_1.mp4')
# y, sr = librosa.load("data/0/selected/audio0_1.mp4", sr=None)

y, sr = librosa.load(path, sr=None)
mfcc = compute_mfcc(y, sr).T
librosa.display.specshow(mfcc, x_axis='time')

mfccSqueezed = mfcc.T.reshape(1, -1)[0]
mfccSqueezed = [i for i in mfccSqueezed]
print(mfccSqueezed)
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

##########################

# crop ###################
# highest_peak_time = 5.4
# cropStart = round((highest_peak_time - 0.2) * sr)
# cropEnd = round((highest_peak_time + 0.6) * sr)
# y = y[cropStart:cropEnd] # trimming data
##########################



# print(len(y)/sr)
avg_amplitude = librosa.stft(y).mean(axis=0) # np array
audio_time = librosa.frames_to_time(range(len(avg_amplitude)), sr=sr)

fft = fft(y, 300)
fft = np.array(fft)
fft = [i for i in fft]
# print(fft)
# print(len(fft))

# audio
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_ylabel('Sound Amplitude (dB)', color="purple")
ax1.plot(fft, 'm-', label="fft Data")
# ax1.dataAnalysis(mfcc, 'm-', label="Sound Data")
ax1.tick_params(axis='y', labelcolor="purple")

# show graph
fig.tight_layout()
plt.title("Sound Data")
plt.show()

# write data
# sf.write('data/0/selected/audio0_1.wav', y, sr, 'PCM_24')