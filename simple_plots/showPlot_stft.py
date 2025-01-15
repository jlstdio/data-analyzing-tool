import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal

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
        y, _ = librosa.load(path, sr=sr)
        y_list.append(y)
        if len(y) > max_length:
            max_length = len(y)

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


def compute_average_stft(y_padded_list, n_fft=2048, hop_length=512, freq_limit=2000, sr=22050):
    """
    여러 오디오 신호의 STFT를 계산하고 평균을 냅니다.

    Parameters:
        y_padded_list (list of np.ndarray): 패딩된 오디오 신호 리스트.
        n_fft (int): FFT 윈도우 크기.
        hop_length (int): 프레임 간 이동 샘플 수.
        freq_limit (int): 표시할 최대 주파수 (Hz).
        sr (int): 샘플링 레이트.

    Returns:
        S_db_avg_limited (np.ndarray): 평균 STFT 스펙트로그램 (dB).
        frequencies_limited (np.ndarray): 제한된 주파수 벡터.
        t_stft (np.ndarray): STFT 시간 축.
    """
    stft_list = []
    for y in y_padded_list:
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)
        stft_list.append(S)

    # 모든 STFT 배열의 최대 프레임 수 찾기
    max_frames = max(S.shape[1] for S in stft_list)
    max_freq_bins = max(S.shape[0] for S in stft_list)

    # STFT 배열을 최대 프레임 수에 맞춰 패딩
    stft_padded = []
    for S in stft_list:
        if S.shape[1] < max_frames:
            pad_width = max_frames - S.shape[1]
            S_padded = np.pad(S, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_padded = S
        stft_padded.append(S_padded)

    # STFT 평균 계산
    stft_avg = np.mean(stft_padded, axis=0)

    # 주파수 제한
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_indices = np.where(frequencies <= freq_limit)[0]
    S_avg_limited = stft_avg[freq_indices, :]

    # 데시벨 변환
    S_db_avg_limited = librosa.amplitude_to_db(S_avg_limited, ref=np.max)

    # 시간 축 계산
    t_stft = librosa.frames_to_time(np.arange(S_avg_limited.shape[1]), sr=sr, hop_length=hop_length)
    frequencies_limited = frequencies[freq_indices]

    return S_db_avg_limited, frequencies_limited, t_stft


def plot_average_rms_and_stft(rms_avg, t_rms, S_db_avg_limited, frequencies_limited, t_stft, y_padded_list):
    """
    평균 RMS와 평균 STFT를 플롯합니다.

    Parameters:
        rms_avg (np.ndarray): 평균 RMS 값.
        t_rms (np.ndarray): RMS 시간 축.
        S_db_avg_limited (np.ndarray): 평균 STFT 스펙트로그램 (dB).
        frequencies_limited (np.ndarray): 제한된 주파수 벡터.
        t_stft (np.ndarray): STFT 시간 축.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    # 평균 데시벨 미터 플롯 (위쪽)
    ax1.plot(t_rms, 20 * np.log10(rms_avg + 1e-6))  # 작은 값을 더해 로그 0 방지
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_title(f'Average Decibel Meter - {root_path} - {trueLocation}')
    ax1.set_xlim([0, t_stft[-1]])
    ax1.set_ylim([-100, 0])  # 일반적인 dB 범위

    # 평균 STFT 스펙트로그램 플롯 (아래쪽)
    img = ax2.imshow(S_db_avg_limited, aspect='auto', origin='lower',
                     extent=[t_stft[0], t_stft[-1], frequencies_limited[0], frequencies_limited[-1]],
                     cmap='magma', interpolation='nearest')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Average STFT (0-2000 Hz)')

    # 색상 막대 추가
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.set_label('dB')

    plt.show()


def run(audio_paths):
    """
    메인 함수: 여러 오디오 파일의 평균 데시벨 미터와 평균 STFT를 플롯합니다.

    Parameters:
        audio_paths (list of str): 오디오 파일 경로 리스트.
    """
    sr = 22050  # 공통 샘플링 레이트
    n_fft = 2048
    hop_length = 512
    freq_limit = 2000  # Hz

    # 1. 오디오 로드 및 패딩
    y_padded_list = load_and_pad_audio(audio_paths, sr=sr)

    # 2. 평균 RMS 계산
    rms_avg, t_rms = compute_average_rms(y_padded_list, hop_length=hop_length)

    # 3. 평균 STFT 계산
    S_db_avg_limited, frequencies_limited, t_stft = compute_average_stft(
        y_padded_list, n_fft=n_fft, hop_length=hop_length, freq_limit=freq_limit, sr=sr)

    # 4. 플롯 생성
    plot_average_rms_and_stft(rms_avg, t_rms, S_db_avg_limited, frequencies_limited, t_stft, y_padded_list)


if __name__ == "__main__":

    root_path = '10-11th heart-sound-joon-9location_locationDifference_sliced'

    trueLocation = '9'

    location = '34'
    fileNum = 0
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)


    location = '35'
    fileNum = 0
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)


    location = '36'
    fileNum = 0
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)


    location = '37'
    fileNum = 0
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)
