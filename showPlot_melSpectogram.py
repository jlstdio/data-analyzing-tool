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


def compute_average_mel_spectrogram(y_padded_list, n_fft=2048, hop_length=512, n_mels=128, fmax=2000, sr=22050):
    """
    여러 오디오 신호의 Mel Spectrogram을 계산하고 평균을 냅니다.

    Parameters:
        y_padded_list (list of np.ndarray): 패딩된 오디오 신호 리스트.
        n_fft (int): FFT 윈도우 크기.
        hop_length (int): 프레임 간 이동 샘플 수.
        n_mels (int): Mel 밴드 수.
        fmax (int): 최대 주파수 (Hz).
        sr (int): 샘플링 레이트.

    Returns:
        S_db_avg_limited (np.ndarray): 평균 Mel Spectrogram (dB).
        mel_frequencies (np.ndarray): Mel 주파수 벡터.
        t_mel (np.ndarray): Mel Spectrogram 시간 축.
    """
    mel_list = []
    for y in y_padded_list:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
        mel_list.append(S)

    # 모든 Mel Spectrogram 배열의 최대 프레임 수 찾기
    max_frames = max(S.shape[1] for S in mel_list)

    # Mel Spectrogram 배열을 최대 프레임 수에 맞춰 패딩
    mel_padded = []
    for S in mel_list:
        if S.shape[1] < max_frames:
            pad_width = max_frames - S.shape[1]
            S_padded = np.pad(S, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_padded = S
        mel_padded.append(S_padded)

    # Mel Spectrogram 평균 계산
    mel_avg = np.mean(mel_padded, axis=0)

    # 데시벨 변환
    S_db_avg_limited = librosa.power_to_db(mel_avg, ref=np.max)

    # 시간 축 계산
    t_mel = librosa.frames_to_time(np.arange(mel_avg.shape[1]), sr=sr, hop_length=hop_length)

    # Mel 주파수 벡터 가져오기
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax)

    return S_db_avg_limited, mel_frequencies, t_mel


def plot_average_rms_and_mel_spectrogram(rms_avg, t_rms, S_db_avg_limited, mel_frequencies, t_mel):
    """
    평균 RMS와 평균 Mel Spectrogram을 플롯합니다.

    Parameters:
        rms_avg (np.ndarray): 평균 RMS 값.
        t_rms (np.ndarray): RMS 시간 축.
        S_db_avg_limited (np.ndarray): 평균 Mel Spectrogram (dB).
        mel_frequencies (np.ndarray): Mel 주파수 벡터.
        t_mel (np.ndarray): Mel Spectrogram 시간 축.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    # 평균 데시벨 미터 플롯 (위쪽)
    ax1.plot(t_rms, 20 * np.log10(rms_avg + 1e-6))  # 작은 값을 더해 로그 0 방지
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_title(f'Average Decibel Meter - {root_path} - {trueLocation}')
    ax1.set_xlim([0, t_mel[-1]])
    ax1.set_ylim([-100, 0])  # 일반적인 dB 범위

    # 평균 Mel Spectrogram 플롯 (아래쪽)
    img = ax2.imshow(S_db_avg_limited, aspect='auto', origin='lower',
                     extent=[t_mel[0], t_mel[-1], mel_frequencies[0], mel_frequencies[-1]],
                     cmap='magma', interpolation='nearest')
    ax2.set_ylabel('Mel Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Average Mel Spectrogram (0-2000 Hz)')

    # 색상 막대 추가
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.set_label('dB')

    plt.show()


def run(audio_paths):
    """
    메인 함수: 여러 오디오 파일의 평균 데시벨 미터와 평균 Mel Spectrogram을 플롯합니다.

    Parameters:
        audio_paths (list of str): 오디오 파일 경로 리스트.
    """
    sr = 22050  # 공통 샘플링 레이트
    n_fft = 2048  # FFT 윈도우 크기
    hop_length = 512  # 프레임 간 이동 샘플 수
    n_mels = 128  # Mel 밴드 수
    fmax = 2000  # 최대 주파수 (Hz)

    # 1. 오디오 로드 및 패딩
    y_padded_list = load_and_pad_audio(audio_paths, sr=sr)

    # 2. 평균 RMS 계산
    rms_avg, t_rms = compute_average_rms(y_padded_list, hop_length=hop_length)

    # 3. 평균 Mel Spectrogram 계산
    S_db_avg_limited, mel_frequencies, t_mel = compute_average_mel_spectrogram(
        y_padded_list, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax, sr=sr)

    # 4. 플롯 생성
    plot_average_rms_and_mel_spectrogram(rms_avg, t_rms, S_db_avg_limited, mel_frequencies, t_mel)


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

