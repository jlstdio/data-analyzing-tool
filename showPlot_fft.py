import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

root_path = ''
trueLocation = ''

def load_and_pad_audio(audio_paths, sr=22050):
    """
    여러 오디오 파일을 로드하고, 가장 긴 오디오 길이에 맞춰 제로 패딩합니다.

    Parameters:
        audio_paths (list of str): 오디오 파일 경로 리스트.
        sr (int): 공통 샘플링 레이트.

    Returns:
        y_padded_list (list of np.ndarray): 제로 패딩된 오디오 신호 리스트.
    """
    y_list = []
    max_length = 0

    # 모든 오디오 파일을 로드하고 최대 길이 찾기
    for path in audio_paths:
        try:
            y, _ = librosa.load(path, sr=sr)
            y_list.append(y)
            if len(y) > max_length:
                max_length = len(y)
        except Exception as e:
            print(f"오디오 파일을 로드하는 중 오류 발생: {path}\n오류 메시지: {e}")

    # 모든 오디오 신호를 최대 길이에 맞춰 패딩
    y_padded_list = []
    for y in y_list:
        if len(y) < max_length:
            y_padded = np.pad(y, (0, max_length - len(y)), mode='constant')
        else:
            y_padded = y
        y_padded_list.append(y_padded)

    return y_padded_list


def compute_average_rms(y_padded_list, hop_length=512):
    """
    여러 오디오 신호의 RMS 값을 계산하고 평균을 냅니다.

    Parameters:
        y_padded_list (list of np.ndarray): 패딩된 오디오 신호 리스트.
        hop_length (int): 프레임 간 이동 샘플 수.

    Returns:
        rms_avg (np.ndarray): 평균 RMS 값.
        t_rms (np.ndarray): RMS 시간 축.
    """
    rms_list = []
    for y in y_padded_list:
        rms = librosa.feature.rms(y=y, hop_length=hop_length).flatten()
        rms_list.append(rms)

    # 모든 RMS 배열의 최대 길이 찾기
    max_frames = max(len(rms) for rms in rms_list)

    # RMS 배열을 최대 길이에 맞춰 패딩
    rms_padded = []
    for rms in rms_list:
        if len(rms) < max_frames:
            rms_padded.append(np.pad(rms, (0, max_frames - len(rms)), mode='constant'))
        else:
            rms_padded.append(rms)

    # 평균 RMS 계산
    rms_avg = np.mean(rms_padded, axis=0)
    t_rms = librosa.frames_to_time(np.arange(len(rms_avg)), hop_length=hop_length, sr=22050)

    return rms_avg, t_rms


def compute_average_fft(y_padded_list, sr=22050, freq_limit=2000):
    """
    여러 오디오 신호의 FFT를 계산하고 평균을 냅니다.

    Parameters:
        y_padded_list (list of np.ndarray): 패딩된 오디오 신호 리스트.
        sr (int): 샘플링 레이트.
        freq_limit (int): 표시할 최대 주파수 (Hz).

    Returns:
        fft_avg_db_limited (np.ndarray): 평균 FFT 스펙트럼 (dB).
        freqs_limited (np.ndarray): 제한된 주파수 벡터.
    """
    fft_magnitudes = []
    for y in y_padded_list:
        # FFT 계산
        N = len(y)
        Y = np.fft.fft(y)
        Y = Y[:N // 2]  # 양수 주파수만 사용
        mag = np.abs(Y) / N  # 정규화된 진폭
        fft_magnitudes.append(mag)

    # 모든 FFT 배열의 최대 길이 찾기
    max_length = max(len(mag) for mag in fft_magnitudes)

    # FFT 배열을 최대 길이에 맞춰 패딩
    fft_padded = []
    for mag in fft_magnitudes:
        if len(mag) < max_length:
            pad_width = max_length - len(mag)
            mag_padded = np.pad(mag, (0, pad_width), mode='constant')
        else:
            mag_padded = mag
        fft_padded.append(mag_padded)

    # FFT 평균 계산
    fft_avg = np.mean(fft_padded, axis=0)

    # 주파수 벡터 계산
    freqs = np.fft.fftfreq(max_length * 2, d=1 / sr)[:max_length]

    # 주파수 제한
    freq_indices = np.where(freqs <= freq_limit)[0]
    freqs_limited = freqs[freq_indices]
    fft_avg_limited = fft_avg[freq_indices]

    # 데시벨 변환
    fft_avg_db_limited = 20 * np.log10(fft_avg_limited + 1e-6)  # 작은 값을 더해 로그 0 방지

    return fft_avg_db_limited, freqs_limited


def plot_average_rms_and_fft(rms_avg, t_rms, fft_avg_db_limited, freqs_limited):
    """
    평균 RMS와 평균 FFT를 플롯합니다.

    Parameters:
        rms_avg (np.ndarray): 평균 RMS 값.
        t_rms (np.ndarray): RMS 시간 축.
        fft_avg_db_limited (np.ndarray): 평균 FFT 스펙트럼 (dB).
        freqs_limited (np.ndarray): 제한된 주파수 벡터.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False, constrained_layout=True)

    # 평균 데시벨 미터 플롯 (위쪽)
    ax1.plot(t_rms, 20 * np.log10(rms_avg + 1e-6))  # 작은 값을 더해 로그 0 방지
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_title(f'Average Decibel Meter - {root_path} - {trueLocation}')
    ax1.set_xlim([0, t_rms[-1]])
    ax1.set_ylim([-100, 0])  # 일반적인 dB 범위
    ax1.grid(True)

    # 평균 FFT 스펙트럼 플롯 (아래쪽)
    ax2.plot(freqs_limited, fft_avg_db_limited, color='m')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Average FFT Spectrum (0-2000 Hz)')
    ax2.set_xlim([0, freqs_limited[-1]])
    ax2.set_ylim([-100, np.max(fft_avg_db_limited) + 10])  # 적절한 dB 범위 설정
    ax2.grid(True)

    plt.show()


def run(audio_paths):
    """
    메인 함수: 여러 오디오 파일의 평균 데시벨 미터와 평균 FFT를 플롯합니다.

    Parameters:
        audio_paths (list of str): 오디오 파일 경로 리스트.
    """
    sr = 22050  # 공통 샘플링 레이트
    hop_length = 512  # RMS 계산을 위한 프레임 간 이동 샘플 수
    freq_limit = 2000  # 최대 주파수 (Hz)

    # 1. 오디오 로드 및 패딩
    y_padded_list = load_and_pad_audio(audio_paths, sr=sr)

    if not y_padded_list:
        print("오디오 파일을 로드할 수 없습니다. 경로를 확인해주세요.")
        return

    # 2. 평균 RMS 계산
    rms_avg, t_rms = compute_average_rms(y_padded_list, hop_length=hop_length)

    # 3. 평균 FFT 계산
    fft_avg_db_limited, freqs_limited = compute_average_fft(
        y_padded_list, sr=sr, freq_limit=freq_limit)

    # 4. 플롯 생성
    plot_average_rms_and_fft(rms_avg, t_rms, fft_avg_db_limited, freqs_limited)


if __name__ == "__main__":

    root_path = '10-11th heart-sound-joon-9location_locationDifference_sliced'
    audio_paths = []

    trueLocation = '1'

    location = '2'
    for fileNum in range(0, 19):
        file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
        audio_paths.append(f'../data/{root_path}/{location}/{file_name}')

    location = '3'
    for fileNum in range(0, 21):
        file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
        audio_paths.append(f'../data/{root_path}/{location}/{file_name}')

    location = '4'
    for fileNum in range(0, 19):
        file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
        audio_paths.append(f'../data/{root_path}/{location}/{file_name}')

    location = '5'
    for fileNum in range(0, 17):
        file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
        audio_paths.append(f'../data/{root_path}/{location}/{file_name}')

    run(audio_paths)
