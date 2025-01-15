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


def compute_average_y(y_padded_list, sr=22050):
    """
    여러 오디오 신호의 평균 y 값을 계산합니다.

    Parameters:
        y_padded_list (list of np.ndarray): 패딩된 오디오 신호 리스트.
        sr (int): 샘플링 레이트.

    Returns:
        y_avg (np.ndarray): 평균 y 값.
        time_avg (np.ndarray): 시간 축.
    """
    # 모든 y 배열을 스택하여 평균 계산
    y_stack = np.vstack(y_padded_list)
    y_avg = np.mean(y_stack, axis=0)

    # 시간 축 계산
    time_avg = np.linspace(0, len(y_avg) / sr, num=len(y_avg))

    return y_avg, time_avg


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
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                           n_mels=n_mels, fmax=fmax, power=2.0)  # power=2.0 for power spectrogram
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


def compute_single_mel_spectrogram(y, n_fft=2048, hop_length=512, n_mels=128, fmax=2000, sr=22050):
    """
    단일 오디오 신호의 Mel Spectrogram을 계산합니다.

    Parameters:
        y (np.ndarray): 오디오 신호.
        n_fft (int): FFT 윈도우 크기.
        hop_length (int): 프레임 간 이동 샘플 수.
        n_mels (int): Mel 밴드 수.
        fmax (int): 최대 주파수 (Hz).
        sr (int): 샘플링 레이트.

    Returns:
        S_db_limited (np.ndarray): Mel Spectrogram (dB).
        mel_frequencies (np.ndarray): Mel 주파수 벡터.
        t_mel (np.ndarray): Mel Spectrogram 시간 축.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, fmax=fmax, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    t_mel = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax)
    return S_db, mel_frequencies, t_mel


def plot_average_y_and_mel_spectrogram(y_avg, time_avg, S_db_avg_limited, mel_frequencies, t_mel):
    """
    평균 y 값과 평균 Mel Spectrogram을 플롯합니다.

    Parameters:
        y_avg (np.ndarray): 평균 y 값.
        time_avg (np.ndarray): 시간 축.
        S_db_avg_limited (np.ndarray): 평균 Mel Spectrogram (dB).
        mel_frequencies (np.ndarray): Mel 주파수 벡터.
        t_mel (np.ndarray): Mel Spectrogram 시간 축.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    # 평균 y 값 플롯 (위쪽)
    ax1.plot(time_avg, y_avg, color='b')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Average Audio Signal - {root_path} - {trueLocation}')
    ax1.set_xlim([0, time_avg[-1]])
    ax1.set_ylim([np.min(y_avg) * 1.1, np.max(y_avg) * 1.1])  # 여유 있는 y축 범위
    ax1.grid(True)

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


def plot_single_y_and_mel_spectrogram(y, time, S_db, mel_frequencies, t_mel):
    """
    단일 y 값과 Mel Spectrogram을 플롯합니다.

    Parameters:
        y (np.ndarray): y 값.
        time (np.ndarray): 시간 축.
        S_db (np.ndarray): Mel Spectrogram (dB).
        mel_frequencies (np.ndarray): Mel 주파수 벡터.
        t_mel (np.ndarray): Mel Spectrogram 시간 축.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    # 단일 y 값 플롯 (위쪽)
    ax1.plot(time, y, color='b')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio Signal - {root_path} - {trueLocation}')
    ax1.set_xlim([0, time[-1]])
    ax1.set_ylim([np.min(y) * 1.1, np.max(y) * 1.1])  # 여유 있는 y축 범위
    ax1.grid(True)

    # Mel Spectrogram 플롯 (아래쪽)
    img = ax2.imshow(S_db, aspect='auto', origin='lower',
                     extent=[t_mel[0], t_mel[-1], mel_frequencies[0], mel_frequencies[-1]],
                     cmap='magma', interpolation='nearest')
    ax2.set_ylabel('Mel Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Mel Spectrogram (0-2000 Hz)')

    # 색상 막대 추가
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.set_label('dB')

    plt.show()


def run(audio_paths):
    """
    메인 함수: 여러 오디오 파일의 평균 y 값과 평균 Mel Spectrogram을 플롯합니다.
    단일 오디오 파일일 경우, 평균을 내지 않고 직접 플롯합니다.

    Parameters:
        audio_paths (list of str): 오디오 파일 경로 리스트.
    """
    sr = 22050  # 공통 샘플링 레이트
    n_fft = 2048  # FFT 윈도우 크기
    hop_length = 64  # 프레임 간 이동 샘플 수
    n_mels = 128  # Mel 밴드 수
    fmax = 2000  # 최대 주파수 (Hz)

    # 1. 오디오 로드 및 패딩
    y_padded_list = load_and_pad_audio(audio_paths, sr=sr)

    if not y_padded_list:
        print("오디오 파일을 로드할 수 없습니다. 경로를 확인해주세요.")
        return

    # 2. 오디오 파일 개수에 따라 처리
    if len(y_padded_list) == 1:
        # 데이터가 1개인 경우
        y = y_padded_list[0]
        time = np.linspace(0, len(y) / sr, num=len(y))

        # Mel Spectrogram 계산
        S_db, mel_frequencies, t_mel = compute_single_mel_spectrogram(y, n_fft=n_fft, hop_length=hop_length,
                                                                      n_mels=n_mels, fmax=fmax, sr=sr)
        # 플롯 생성
        plot_single_y_and_mel_spectrogram(y, time, S_db, mel_frequencies, t_mel)
    else:
        # 데이터가 여러 개인 경우 평균 계산
        y_avg, time_avg = compute_average_y(y_padded_list, sr=sr)
        S_db_avg_limited, mel_frequencies, t_mel = compute_average_mel_spectrogram(
            y_padded_list, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax, sr=sr)
        # 플롯 생성
        plot_average_y_and_mel_spectrogram(y_avg, time_avg, S_db_avg_limited, mel_frequencies, t_mel)


if __name__ == "__main__":
    root_path = '10-11th heart-sound-joon-9location_locationDifference_sliced'

    trueLocation = '1'
    location = '5'  # 2,3,4,5
    fileNum = 10
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)

    trueLocation = '2'
    location = '9'  # 6,7,8,9
    fileNum = 10
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)

    trueLocation = '8'
    location = '33'  # 30, 31, 32, 33
    fileNum = 10
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)

    trueLocation = '9'
    location = '37'  # 34, 35, 36, 37
    fileNum = 10
    file_name = f'user20241011-location1-data{location}-audioData_{fileNum}.wav'
    audio_paths = [f'../data/{root_path}/{location}/{file_name}']
    run(audio_paths)
