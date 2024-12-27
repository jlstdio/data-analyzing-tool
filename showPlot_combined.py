import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from util import *

# load data - acc
data = pd.read_csv('../data/0/selected/acc0_1.csv')
# time resolution : 25.6sec = 2566 frame -> 1/100

startTime = 5.8
cropRange = 0.6
# data = crop_accelerometer_data(data, startTime, startTime + (cropRange))

# acc data extraction
time = data['time'].values
accel_x = data['x'].values
accel_y = data['y'].values
accel_z = data['z'].values

# audio data init
y, sr = librosa.load("../data/0/selected/audio0_1.mp4", sr=None)
# y = y[round(1*sr):round(4*sr)] # num matric : second
avg_amplitude = librosa.stft(y).mean(axis=0) # np array
audio_time = librosa.frames_to_time(range(len(avg_amplitude)), sr=sr) # np array

# graph init
fig, ax1 = plt.subplots(figsize=(12, 6))

# acc
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Accelerometer Data', color="blue")
# ax1.dataAnalysis(time, accel_x, 'b-', label="X-axis")
# ax1.dataAnalysis(time, accel_y, 'g-', label="Y-axis")
ax1.plot(time, accel_z, 'r-', label="Z-axis")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.legend(loc="upper left")

# audio

ax2 = ax1.twinx()
ax2.set_ylabel('Sound Amplitude (dB)', color="purple")
ax2.plot(audio_time, avg_amplitude, 'm-', label="Sound Data")
ax2.tick_params(axis='y', labelcolor="purple")


# show graph
fig.tight_layout()
plt.title("Accelerometer and Sound Data vs. Time")
plt.show()