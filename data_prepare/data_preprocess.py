import json
import os
import numpy as np
import pandas as pd
import torch
import librosa  # STFT를 위해 추가
from scipy.io import wavfile
from torch.utils.data import DataLoader, TensorDataset


def standarize(data):
    return (data - data.mean()) / data.std()


def prepareData_accWithSTFT_2d(root, folderPathList, batchSize, mode='train', train_ratio=0.9, n_fft=2048, hop_length=512, max_freq=2000):
    combinedData = []
    labels = []

    stft_lengths = []
    frequency_mask_cache = {}  # 샘플링 레이트별로 주파수 마스크를 캐싱하기 위해 사용
    max_time_steps = 0
    fixed_freq_bins = None  # 주파수 축 크기를 고정

    # 첫 번째 패스: 모든 STFT 길이를 수집하여 최대 시간 축을 찾고 주파수 축을 고정합니다.
    for folderPath in folderPathList:

        with open(f"{root}/{folderPath}/positionMap.json", 'r') as file:
            locationMap_train = json.load(file)

        userNumber = int(locationMap_train['userId'])
        labelMap= locationMap_train[f'position_{mode}']

        for dataNum in labelMap.keys():
            dataNum = int(dataNum)
            accel_fileFormat = f'user{userNumber}-location1-data{dataNum}-accelData'
            audio_fileFormat = f'user{userNumber}-location1-data{dataNum}-audioData'

            lst = os.listdir(f'{root}/{folderPath}/{dataNum}/')
            filesNum = int(len(lst) / 2)

            print(f'from label {dataNum} : {filesNum} files in data {dataNum}')

            for i in range(filesNum):
                rawAccelPath = f'{root}/{folderPath}/{dataNum}/{accel_fileFormat}_{i}.csv'
                rawAudioPath = f'{root}/{folderPath}/{dataNum}/{audio_fileFormat}_{i}.wav'

                # 데이터 로드 - accel & audio
                rawAccelData = pd.read_csv(rawAccelPath)
                fs, audio_data = wavfile.read(rawAudioPath)

                # 오디오 데이터를 부동 소수점으로 변환
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif audio_data.dtype in [np.float32, np.float64]:
                    audio_data = audio_data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

                # STFT 적용
                stftData = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
                stft_magnitude = np.abs(stftData)

                # 주파수 제한 적용
                if fs not in frequency_mask_cache:
                    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
                    frequency_mask = freqs <= max_freq
                    if not np.any(frequency_mask):
                        raise ValueError(f"No frequencies below {max_freq}Hz for sampling rate {fs} and n_fft {n_fft}.")
                    frequency_mask_cache[fs] = frequency_mask
                else:
                    frequency_mask = frequency_mask_cache[fs]

                stft_magnitude = stft_magnitude[frequency_mask, :]

                # 주파수 제한 후 STFT 데이터가 비어있는지 확인
                if stft_magnitude.size == 0:
                    print(f'STFT magnitude is empty after frequency masking for file {rawAudioPath}. Skipping.')
                    continue

                # STFT의 주파수 축 크기를 고정
                if fixed_freq_bins is None:
                    fixed_freq_bins = stft_magnitude.shape[0]
                else:
                    if stft_magnitude.shape[0] != fixed_freq_bins:
                        print(f'Inconsistent frequency bins in file {rawAudioPath}. Skipping.')
                        continue

                T = stft_magnitude.shape[1]
                if T > max_time_steps:
                    max_time_steps = T

                # 가속도 데이터는 80으로 고정
                if 80 > max_time_steps:
                    max_time_steps = 80

    print(f"Maximum time steps determined: {max_time_steps}")
    print(f"Fixed frequency bins: {fixed_freq_bins}")

    # 두 번째 패스: 데이터를 4채널으로 구성하고 패딩합니다.
    for folderPath in folderPathList:

        with open(f"{root}/{folderPath}/positionMap.json", 'r') as file:
            locationMap_train = json.load(file)

        userNumber = int(locationMap_train['userId'])
        labelMap= locationMap_train[f'position_{mode}']

        for dataNum in labelMap.keys():
            dataNum = int(dataNum)
            accel_fileFormat = f'user{userNumber}-location1-data{dataNum}-accelData'
            audio_fileFormat = f'user{userNumber}-location1-data{dataNum}-audioData'

            lst = os.listdir(f'{root}/{folderPath}/{dataNum}/')
            filesNum = int(len(lst) / 2)

            print(f'from label {dataNum} : {filesNum} files in data {dataNum}')

            for i in range(filesNum):
                rawAccelPath = f'{root}/{folderPath}/{dataNum}/{accel_fileFormat}_{i}.csv'
                rawAudioPath = f'{root}/{folderPath}/{dataNum}/{audio_fileFormat}_{i}.wav'

                # 데이터 로드 - accel & audio
                rawAccelData = pd.read_csv(rawAccelPath)
                fs, audio_data = wavfile.read(rawAudioPath)

                # 오디오 데이터를 부동 소수점으로 변환
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif audio_data.dtype in [np.float32, np.float64]:
                    audio_data = audio_data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

                # STFT 적용
                stftData = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
                stft_magnitude = np.abs(stftData)

                # 주파수 제한 적용
                frequency_mask = frequency_mask_cache[fs]
                stft_magnitude = stft_magnitude[frequency_mask, :]

                # 주파수 제한 후 STFT 데이터가 비어있는지 확인
                if stft_magnitude.size == 0:
                    print(f'STFT magnitude is empty after frequency masking for file {rawAudioPath}. Skipping.')
                    continue

                # STFT의 주파수 축 크기가 고정되었는지 확인
                if stft_magnitude.shape[0] != fixed_freq_bins:
                    print(f'Inconsistent frequency bins in file {rawAudioPath}. Skipping.')
                    continue

                ## if data returns numbers
                acc_x = standarize(rawAccelData["accel_x"][:80]).values
                acc_y = standarize(rawAccelData["accel_y"][:80]).values
                acc_z = standarize(rawAccelData["accel_z"][:80]).values

                if len(acc_x) != 80:
                    print(f'{rawAccelPath} - accel_x')
                    continue
                if len(acc_y) != 80:
                    print(f'{rawAccelPath} - accel_y')
                    continue
                if len(acc_z) != 80:
                    print(f'{rawAccelPath} - accel_z')
                    continue

                # 입력 데이터에 NaN 또는 무한대 값이 있는지 확인
                if np.any(np.isnan(acc_x)) or np.any(np.isnan(acc_y)) or np.any(np.isnan(acc_z)):
                    print(f'NaN detected in dataNum {dataNum}, file {i}')
                    continue
                if np.any(np.isinf(acc_x)) or np.any(np.isinf(acc_y)) or np.any(np.isinf(acc_z)):
                    print(f'Infinite value detected in dataNum {dataNum}, file {i}')
                    continue

                # STFT를 표준화하고 패딩합니다.
                stft_magnitude_df = pd.DataFrame(stft_magnitude)
                audio_stft = standarize(stft_magnitude_df).values  # Shape: (F, T)
                T = audio_stft.shape[1]

                if T > max_time_steps:
                    audio_stft_padded = audio_stft[:, :max_time_steps]
                else:
                    pad_width = max_time_steps - T
                    audio_stft_padded = np.pad(audio_stft, ((0, 0), (0, pad_width)), 'constant')

                # 가속도 데이터를 패딩합니다.
                pad_length_acc = max_time_steps - 80
                if pad_length_acc < 0:
                    # 가속도 데이터가 길면 자릅니다.
                    acc_x_padded = acc_x[:max_time_steps]
                    acc_y_padded = acc_y[:max_time_steps]
                    acc_z_padded = acc_z[:max_time_steps]
                else:
                    acc_x_padded = np.pad(acc_x, (0, pad_length_acc), 'constant')
                    acc_y_padded = np.pad(acc_y, (0, pad_length_acc), 'constant')
                    acc_z_padded = np.pad(acc_z, (0, pad_length_acc), 'constant')

                # 가속도 데이터를 2D로 변환 (F, T) 형태로 만듭니다.
                # 여기서는 F=1로 가정하고, F=1을 고정 주파수 축으로 사용
                # 이후 CNN에서 주파수 축과 동일하게 맞추기 위해 F를 fixed_freq_bins로 확장합니다.
                acc_x_2d = np.tile(acc_x_padded, (fixed_freq_bins, 1))  # Shape: (F, T)
                acc_y_2d = np.tile(acc_y_padded, (fixed_freq_bins, 1))
                acc_z_2d = np.tile(acc_z_padded, (fixed_freq_bins, 1))

                # 모든 배열의 크기를 확인 (디버깅용)
                assert audio_stft_padded.shape == (fixed_freq_bins, max_time_steps), f'audio_stft_padded shape mismatch: {audio_stft_padded.shape}'
                assert acc_x_2d.shape == (fixed_freq_bins, max_time_steps), f'acc_x_2d shape mismatch: {acc_x_2d.shape}'
                assert acc_y_2d.shape == (fixed_freq_bins, max_time_steps), f'acc_y_2d shape mismatch: {acc_y_2d.shape}'
                assert acc_z_2d.shape == (fixed_freq_bins, max_time_steps), f'acc_z_2d shape mismatch: {acc_z_2d.shape}'

                # 4채널으로 스택합니다. (채널, F, T)
                sample = np.stack([audio_stft_padded, acc_x_2d, acc_y_2d, acc_z_2d], axis=0)
                combinedData.append(sample)

                labels.append(labelMap[str(dataNum)])

    combinedData = np.array(combinedData)  # Shape: (num_samples, 4, F, T)
    labels = np.array(labels)

    print(f"Combined data shape: {combinedData.shape}")  # (num_samples, 4, F, T)
    print(f"Labels shape: {labels.shape}")

    # 데이터 섞기
    indices = np.arange(len(combinedData))
    np.random.seed(42)
    np.random.shuffle(indices)
    combinedData = combinedData[indices]
    labels = labels[indices]

    train_end = int(train_ratio * len(combinedData))

    train_data = combinedData[:train_end]
    train_labels = labels[:train_end]

    print(f"Train data shape: {train_data.shape}")  # (N * 0.9, 4, F, T)

    # X_train = torch.tensor(train_data, dtype=torch.float32)
    # y_train = torch.tensor(train_labels, dtype=torch.long)

    X_train = train_data
    y_train = train_labels
    
    # train_dataset = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    '''
    # val_loader = None
    X_val = None
    y_val = None

    # 검증 데이터 생성
    if train_ratio != 1.0:
        validation_data = combinedData[train_end:]
        validation_labels = labels[train_end:]

        print(f"Validation data shape: {validation_data.shape}")  # (N * 0.1, 4, F, T)

        X_val = torch.tensor(validation_data, dtype=torch.float32)
        y_val = torch.tensor(validation_labels, dtype=torch.long)

        # val_dataset = TensorDataset(X_val, y_val)
        # val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
    '''

    # return (X_train, y_train), (X_val, y_val)
    return X_train, y_train
