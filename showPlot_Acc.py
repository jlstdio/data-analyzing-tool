import matplotlib.pyplot as plt
import pandas as pd
from util.featureExtractor import *

def showFeatures(path):
    # load data - acc
    data = pd.read_csv(path)
    time = data['Time'].values
    time -= time[0]

    # acc data extraction
    accel_x = data['accel_x'].values  # 전화기 디스플레이 기준으로 수평으로 위 아래
    accel_y = data['accel_y'].values  # 전화기 새웠을 때 세로
    accel_z = data['accel_z'].values  # 전화기 새웠을 때 가로
    sum_acc = accel_x + accel_y + accel_z

    # graph init
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    # acc
    ax1.set_xlabel('time')
    ax1.set_ylabel('acc')
    ax1.plot(time, accel_x, 'b-', label="X-axis")
    ax1.plot(time, accel_y, 'g-', label="Y-axis")
    ax1.plot(time, accel_z, 'r-', label="Z-axis")
    ax1.legend(loc="lower right")
    print(f' raw acc len : {len(accel_x)}')

    ax2.set_xlabel('frequency')
    ax2.set_ylabel('density')
    zcr_x = zero_crossing_rate(accel_x)
    ax2.plot(zcr_x, 'b-', label="zcr x")
    zcr_y = zero_crossing_rate(accel_y)
    ax2.plot(zcr_y, 'g-', label="zcr y")
    zcr_z = zero_crossing_rate(accel_z)
    ax2.plot(zcr_z, 'r-', label="zcr z")
    ax2.legend(loc="lower right")
    # print(f' zero crossing rate : {len(zcr_x)}')

    '''
    # psd
    ax2.set_xlabel('frequency')
    ax2.set_ylabel('density')
    f, S = psd(accel_x)
    print(len(f))
    ax2.dataAnalysis(f, S, 'b-', label="psd x")
    f, S = psd(accel_y)
    ax2.dataAnalysis(f, S, 'g-', label="psd y")
    f, S = psd(accel_z)
    ax2.dataAnalysis(f, S, 'r-', label="psd z")
    ax2.legend(loc="lower right")
    print(f' psd : {len(S)}')
    '''

    # fft
    ax3.set_xlabel('frequency')
    ax3.set_ylabel('power')
    f = fft(accel_x)
    ax3.plot(f, 'b-', label="fft x")
    f = fft(accel_y)
    ax3.plot(f, 'g-', label="fft y")
    f = fft(accel_z)
    ax3.plot(f, 'r-', label="fft z")
    ax3.legend(loc="lower right")
    # print(f' fft : {len(f)}')

    # auto-correlation
    ax4.set_xlabel('frequency')
    ax4.set_ylabel('power')
    f = [acf(accel_x, k) for k in range(20)]
    ax4.plot(f, 'b-', label="autocorr x")
    f = [acf(accel_y, k) for k in range(20)]
    ax4.plot(f, 'g-', label="autocorr y")
    f = [acf(accel_z, k) for k in range(20)]
    ax4.plot(f, 'r-', label="autocorr z")
    ax4.legend(loc="lower right")
    # print(f' acf : {len(f)}')

    # show graph
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    plt.title("Feature")
    plt.show()


def showRawData(dataDir, preview=True):
    # load data - acc
    data = pd.read_csv(dataDir)
    time = data['Time'].values
    time -= time[0]
    if preview:
        # data = crop_accelerometer_data(data, 5.7 * 10 ** 9, 13.7 * 10 ** 9)
        time = data['Time'].values.astype('float64')  # in nanoseconds
        time -= time[0]
        time /= 1000000000.0  # 1000000000ns = 1sec

    # acc data extraction
    accel_x = data['accel_x'].values  # 전화기 디스플레이 기준으로 수평으로 위 아래
    accel_y = data['accel_y'].values  # 전화기 새웠을 때 세로
    accel_z = data['accel_z'].values  # 전화기 새웠을 때 가로
    sum_acc = accel_x + accel_y + accel_z

    index_peak_x, value_peak_x = None, None
    index_peak_y, value_peak_y = None, None
    index_peak_z, value_peak_z = None, None
    # index_peak_x, value_peak_x = highest_peak(data=accel_x, time=time, start_time=0, end_time=5)
    # index_peak_y, value_peak_y = highest_peak(data=accel_y, time=time, start_time=0, end_time=5)
    # index_peak_z, value_peak_z = highest_peak(data=accel_z, time=time,  start_time=0, end_time=5)

    # graph init
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # acc
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Accelerometer Data')

    # ax1.dataAnalysis(time, accel_x, 'b-', label="X-axis")
    # ax1.dataAnalysis(time, accel_y, 'g-', label="Y-axis")
    ax1.plot(time, accel_z, 'r-', label="Z-axis")
    # ax1.dataAnalysis(time, sum_acc, 'b-', label="sum-axis")

    if index_peak_x is not None:
        plt.plot(time[index_peak_x], value_peak_x, 'bo', label=f'Highest Peak = {value_peak_x}')

    if index_peak_y is not None:
        plt.plot(time[index_peak_y], value_peak_y, 'go', label=f'Highest Peak = {value_peak_y}')

    if index_peak_z is not None:
        plt.plot(time[index_peak_z], value_peak_z, 'ro', label=f'Highest Peak = {value_peak_z}')

    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.legend(loc="lower right")

    # show graph
    fig.tight_layout()

    # 눈금 간격 1로 변경(x, y축 모두)
    # plt.xticks(range(27))
    # plt.grid(axis='x')

    # Define custom grid intervals
    time_interval = 1.0  # for example, every 0.5 units on the x-axis
    accel_interval = 1.0  # for example, every 1.0 units on the y-axis
    # plt.axvline(time[index_peak_x], color='g', linestyle='--')
    plt.xticks(np.arange(0.0, max(time) + time_interval, time_interval))
    plt.grid(axis='x')

    plt.title("Accelerometer")
    plt.show()


# showing raw data
'''
for i in range(820):
    print(str(i) + " : ", end='')
    path = 'data/splitData/acc8/acc8_' + str(i) + '.csv'
    data = pd.read_csv(path)
    data = crop_accelerometer_data(data, 0, 1 * 10 ** 9)
    time = data['time'].values
    print(len(time))
'''

# showing features
showFeatures('../../data/11-25th heart-sound-joon-fineLocation_raw_sliced/10/data10-accelData_11.csv')

'''
for i in range(0, 200, 5):
    data = pd.read_csv('data/splitData/acc8/acc8_' + str(i) + '.csv')
    time = data['time'].values
    time -= time[0]

    # acc data extraction
    accel_x = data['x'].values  # 전화기 디스플레이 기준으로 수평으로 위 아래
    accel_y = data['y'].values  # 전화기 새웠을 때 세로
    accel_z = data['z'].values  # 전화기 새웠을 때 가로

    zcr_x = zero_crossing_rate(accel_x)
    zcr_y = zero_crossing_rate(accel_y)
    zcr_z = zero_crossing_rate(accel_z)

    csd_x = consecutive_diff(accel_x)
    csd_y = consecutive_diff(accel_y)
    csd_z = consecutive_diff(accel_z)

    print(f'{zcr_x} {zcr_y} {zcr_z}')
'''