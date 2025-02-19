import os
import shutil
import tarfile
import urllib.request
import pickle
import numpy as np

from scipy.ndimage import rotate
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import matplotlib
import matplotlib.pyplot as plt
   

class cifar10_expanded_3_dataloader(object):
    CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    ARCHIVE_NAME = 'cifar-10-python.tar.gz'

    def __init__(self, data_dir, normalize=True):
        self.data_dir = data_dir
        self.batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        self.test_file = 'test_batch'
        self.normalize = normalize

        # 내부적으로 저장할 원본 데이터 (정규화 전/후)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # 정규화 통계
        self.mean = None
        self.std = None

        # 폴더 생성
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}")

        # 파일 체크 후, 없으면 다운로드/추출
        if not self._check_files_exist():
            print("Required CIFAR-10 files not found. Downloading dataset...")
            self._download_and_extract()
            print("Download and extraction complete.")
        else:
            print("All CIFAR-10 files are present.")

    def _check_files_exist(self):
        for batch_file in self.batch_files + [self.test_file]:
            file_path = os.path.join(self.data_dir, batch_file)
            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                return False
        return True

    def _download_and_extract(self):
        archive_path = os.path.join(self.data_dir, self.ARCHIVE_NAME)

        # 다운로드
        print(f"Downloading CIFAR-10 dataset from {self.CIFAR10_URL}...")
        try:
            urllib.request.urlretrieve(self.CIFAR10_URL,
                                       archive_path,
                                       self._download_progress)
            print("\nDownload finished.")
        except Exception as e:
            raise RuntimeError(f"Failed to download CIFAR-10 dataset: {e}")

        # 압축 해제
        print("Extracting the dataset...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            print("Extraction complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract CIFAR-10 dataset: {e}")
        finally:
            # tar.gz 삭제
            if os.path.exists(archive_path):
                os.remove(archive_path)
                print(f"Removed archive {archive_path}")

            # 추출된 폴더 내 파일 이동
            extracted_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
            if os.path.isdir(extracted_dir):
                for filename in os.listdir(extracted_dir):
                    shutil.move(os.path.join(extracted_dir, filename),
                                self.data_dir)
                os.rmdir(extracted_dir)
                print(f"Moved files from {extracted_dir} to {self.data_dir}")

    def _download_progress(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100
        percent = min(100, percent)
        print(f"\rDownload progress: {percent:.2f}%", end='')

    def _read_batch(self, file):
        """한 개의 batch 파일(또는 test_batch)을 읽어서 return"""
        with open(os.path.join(self.data_dir, file), 'rb') as f:
            dict_ = pickle.load(f, encoding='bytes')
        data = dict_[b'data']
        labels = dict_[b'labels']
        # (N, 3, 32, 32) -> (N, 32, 32, 3)
        data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        return data, np.array(labels)

    def load_data(self):
        """
        CIFAR-10 데이터를 읽어 (x_train, y_train), (x_test, y_test)에 할당.
        만약 normalize=True 인 경우, 내부적으로 mean, std 를 구해 normalize 함.
        """
        x_train_list, y_train_list = [], []
        # 5개의 train batch
        for batch_file in self.batch_files:
            data, labels = self._read_batch(batch_file)
            x_train_list.append(data)
            y_train_list.append(labels)

        self.x_train = np.concatenate(x_train_list, axis=0)
        self.y_train = np.concatenate(y_train_list, axis=0)

        self.x_test, self.y_test = self._read_batch(self.test_file)

        # 정규화
        if self.normalize:
            self.mean = np.mean(self.x_train, axis=(0, 1, 2), keepdims=True)
            self.std = np.std(self.x_train, axis=(0, 1, 2), keepdims=True)
            self.x_train = (self.x_train - self.mean) / (self.std + 1e-7)
            self.x_test = (self.x_test - self.mean) / (self.std + 1e-7)

        print("CIFAR-10 loaded.")
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def get_mean_std(self):
        """정규화 통계량 반환"""
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been computed. Please call load_data() first.")
        return self.mean, self.std

    # ---------------------- (1) Rotate: 4가지 각도 범위 ----------------------
    def _random_rotate_range(self, images, min_angle=-30, max_angle=30):
        """
        지정된 각도 범위(min_angle ~ max_angle) 내에서 랜덤 각도를 선택해 회전.
        images shape: (N, 32, 32, 3)
        """
        N = images.shape[0]
        images_rotated = np.zeros_like(images)
        for i in range(N):
            angle = np.random.uniform(min_angle, max_angle)
            rotated_img = rotate(images[i], angle, reshape=False, axes=(0, 1))
            images_rotated[i] = rotated_img
        return images_rotated

    def get_rotate_data_1(self):
        """-30 ~ +30도"""
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_rot = self._random_rotate_range(self.x_train, -10, 10)
        x_test_rot = self._random_rotate_range(self.x_test, -10, 10)
        return (x_train_rot, self.y_train.copy()), (x_test_rot, self.y_test.copy())

    def get_rotate_data_2(self):
        """60 ~ 120도"""
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_rot = self._random_rotate_range(self.x_train, 80, 100)
        x_test_rot = self._random_rotate_range(self.x_test, 80, 100)
        return (x_train_rot, self.y_train.copy()), (x_test_rot, self.y_test.copy())

    def get_rotate_data_3(self):
        """150 ~ 210도"""
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_rot = self._random_rotate_range(self.x_train, 170, 190)
        x_test_rot = self._random_rotate_range(self.x_test, 170, 190)
        return (x_train_rot, self.y_train.copy()), (x_test_rot, self.y_test.copy())

    def get_rotate_data_4(self):
        """240 ~ 310도"""
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_rot = self._random_rotate_range(self.x_train, 260, 280)
        x_test_rot = self._random_rotate_range(self.x_test, 260, 280)
        return (x_train_rot, self.y_train.copy()), (x_test_rot, self.y_test.copy())

    # ---------------------- (2) Color Emphasis (빨강/초록/주황/보라) ----------------------
    def _color_emphasis(self, images, mode='red'):
        """
        특정 색감을 강조하기 위한 간단한 채널별 스케일링 예시.
          - red     : 빨강 채널을 크게
          - green   : 초록 채널을 크게
          - orange  : 빨강/초록을 함께 키우되, blue는 줄임
          - purple  : 빨강/파랑을 키우고, green은 줄임
        """
        images = images.copy()
        N = images.shape[0]

        if mode == 'red':
            # R 채널만 좀 더 키우기
            scale = np.array([1.2, 1.0, 1.0])
        elif mode == 'green':
            scale = np.array([1.0, 1.2, 1.0])
        elif mode == 'orange':
            # 주황색: 빨강+초록 계열
            scale = np.array([1.2, 1.15, 0.9])
        elif mode == 'purple':
            # 보라색: 빨강+파랑 계열
            scale = np.array([1.1, 0.9, 1.1])
        else:
            # 기본값은 변경없음
            scale = np.array([1.0, 1.0, 1.0])

        # 채널별 스케일링
        for i in range(N):
            images[i] = images[i] * scale

        return images

    def get_jitter_data_red(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_jit = self._color_emphasis(self.x_train, 'red')
        x_test_jit = self._color_emphasis(self.x_test, 'red')
        return (x_train_jit, self.y_train.copy()), (x_test_jit, self.y_test.copy())

    def get_jitter_data_green(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_jit = self._color_emphasis(self.x_train, 'green')
        x_test_jit = self._color_emphasis(self.x_test, 'green')
        return (x_train_jit, self.y_train.copy()), (x_test_jit, self.y_test.copy())

    def get_jitter_data_orange(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_jit = self._color_emphasis(self.x_train, 'orange')
        x_test_jit = self._color_emphasis(self.x_test, 'orange')
        return (x_train_jit, self.y_train.copy()), (x_test_jit, self.y_test.copy())

    def get_jitter_data_purple(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_jit = self._color_emphasis(self.x_train, 'purple')
        x_test_jit = self._color_emphasis(self.x_test, 'purple')
        return (x_train_jit, self.y_train.copy()), (x_test_jit, self.y_test.copy())

    # ---------------------- (3) Frequency-based Noise (Filtering) ----------------------
    def _apply_frequency_filter(self, images, filter_type='lowpass'):
        """
        2D FFT -> frequency filter -> IFFT
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        """
        images = images.copy()
        N = images.shape[0]
        H, W, C = images.shape[1], images.shape[2], images.shape[3]  # 32, 32, 3

        # 필터 파라미터 (32x32 기준)
        radius1 = 3   # 대략 저주파/고주파 경계
        radius2 = 7  # bandpass/bandstop 범위

        # 출력 배열
        filtered_images = np.zeros_like(images)

        for i in range(N):
            for c in range(C):
                # 1) FFT
                freq = fft2(images[i, :, :, c])
                freq_shifted = fftshift(freq)

                # 좌표 망 준비
                crow, ccol = H // 2, W // 2
                mask = np.ones((H, W), dtype=np.float32)

                # 거리 계산용
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)

                if filter_type == 'lowpass':
                    # radius1 이하만 통과, 나머지 0
                    mask[dist_from_center > radius1] = 0.0

                elif filter_type == 'highpass':
                    # radius1 이하 0, 나머지 통과
                    mask[dist_from_center < radius1] = 0.0

                elif filter_type == 'bandpass':
                    # radius1 ~ radius2 범위만 통과
                    mask[(dist_from_center < radius1) | (dist_from_center > radius2)] = 0.0

                elif filter_type == 'bandstop':
                    # radius1 ~ radius2 범위만 제외, 나머지 통과
                    mask[(dist_from_center > radius1) & (dist_from_center < radius2)] = 0.0

                # 2) 마스크 적용
                freq_filtered = freq_shifted * mask

                # 3) 역shift, 역FFT
                freq_ishift = ifftshift(freq_filtered)
                img_filtered = ifft2(freq_ishift)
                filtered_images[i, :, :, c] = np.real(img_filtered)

        return filtered_images

    def get_freq_lowpass(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_f = self._apply_frequency_filter(self.x_train, 'lowpass')
        x_test_f = self._apply_frequency_filter(self.x_test, 'lowpass')
        return (x_train_f, self.y_train.copy()), (x_test_f, self.y_test.copy())

    def get_freq_highpass(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_f = self._apply_frequency_filter(self.x_train, 'highpass')
        x_test_f = self._apply_frequency_filter(self.x_test, 'highpass')
        return (x_train_f, self.y_train.copy()), (x_test_f, self.y_test.copy())

    def get_freq_bandpass(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_f = self._apply_frequency_filter(self.x_train, 'bandpass')
        x_test_f = self._apply_frequency_filter(self.x_test, 'bandpass')
        return (x_train_f, self.y_train.copy()), (x_test_f, self.y_test.copy())

    def get_freq_bandstop(self):
        if self.x_train is None:
            raise ValueError("Please call load_data() first.")
        x_train_f = self._apply_frequency_filter(self.x_train, 'bandstop')
        x_test_f = self._apply_frequency_filter(self.x_test, 'bandstop')
        return (x_train_f, self.y_train.copy()), (x_test_f, self.y_test.copy())

    # ---------------------- denormalize 함수 ----------------------
    def denormalize(self, x):
        """
        (x - mean)/std 형태로 정규화된 데이터를 다시 0~255 범위로 복원
        """
        if self.mean is None or self.std is None:
            raise ValueError("Please call load_data() first to compute mean/std.")

        x = x * (self.std + 1e-7) + self.mean  # 역정규화
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    # ---------------------- batch 파일 저장/로드 함수 ----------------------
    def save_dataset_as_batches(self, x_train, y_train, x_test, y_test,
                                version_name, batch_size=10000):
        """
        (x_train, y_train), (x_test, y_test)를 여러 batch 파일로 나눠 저장.
        예: version_name = 'cifar10_original' 이라면,
            ./[data_dir]/cifar10_original/train_batch_0.pkl ...
            ./[data_dir]/cifar10_original/test_batch.pkl
        """
        # version별 폴더 생성
        save_dir = os.path.join(self.data_dir, version_name)
        os.makedirs(save_dir, exist_ok=True)

        # (1) train 세트 저장
        num_train = len(x_train)
        num_train_batches = int(np.ceil(num_train / batch_size))

        for i in range(num_train_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, num_train)
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            filename = os.path.join(save_dir, f"train_batch_{i}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump((x_batch, y_batch), f)
            print(f"Saved {filename}  (shape={x_batch.shape})")

        # (2) test 세트 저장 (보통 하나면 충분하므로 일괄 저장)
        test_filename = os.path.join(save_dir, "test_batch.pkl")
        with open(test_filename, 'wb') as f:
            pickle.dump((x_test, y_test), f)
        print(f"Saved {test_filename}  (shape={x_test.shape})")

    def load_dataset_from_batches(self, version_name):
        """
        save_dataset_as_batches()로 저장된 batch 파일들을 불러와
        (x_train, y_train), (x_test, y_test) 형태로 반환.
        """
        load_dir = os.path.join(self.data_dir, version_name)
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"{load_dir} does not exist.")

        # train batch 파일들 로드
        x_train_list = []
        y_train_list = []

        train_files = sorted([
            f for f in os.listdir(load_dir)
            if f.startswith("train_batch_") and f.endswith(".pkl")
        ])
        if len(train_files) == 0:
            raise FileNotFoundError(f"No train_batch_*.pkl found in {load_dir}")

        for tf in train_files:
            file_path = os.path.join(load_dir, tf)
            with open(file_path, 'rb') as f:
                x_batch, y_batch = pickle.load(f)
            x_train_list.append(x_batch)
            y_train_list.append(y_batch)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # test_batch.pkl 로드
        test_file = os.path.join(load_dir, "test_batch.pkl")
        if not os.path.isfile(test_file):
            raise FileNotFoundError(f"{test_file} does not exist.")

        with open(test_file, 'rb') as f:
            x_test, y_test = pickle.load(f)

        return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # 1) 원본 데이터 로드
    dataloader = cifar10_expanded_3_dataloader('./cifar10_expanded_3', normalize=True)
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    # -----------------------
    # 2) Original 데이터 저장
    # -----------------------

    dataloader.save_dataset_as_batches(x_train, y_train, x_test, y_test,
                                       version_name='cifar10_original',
                                       batch_size=10000)

    # -----------------------
    # 3) Rotate 데이터 4가지
    # -----------------------

    # (1) -30~+30
    (x_train_r1, y_train_r1), (x_test_r1, y_test_r1) = dataloader.get_rotate_data_1()
    dataloader.save_dataset_as_batches(x_train_r1, y_train_r1, x_test_r1, y_test_r1,
                                       version_name='cifar10_rotate_1',
                                       batch_size=10000)

    # (2) 60~120
    (x_train_r2, y_train_r2), (x_test_r2, y_test_r2) = dataloader.get_rotate_data_2()
    dataloader.save_dataset_as_batches(x_train_r2, y_train_r2, x_test_r2, y_test_r2,
                                       version_name='cifar10_rotate_2',
                                       batch_size=10000)

    # (3) 150~210
    (x_train_r3, y_train_r3), (x_test_r3, y_test_r3) = dataloader.get_rotate_data_3()
    dataloader.save_dataset_as_batches(x_train_r3, y_train_r3, x_test_r3, y_test_r3,
                                       version_name='cifar10_rotate_3',
                                       batch_size=10000)

    # (4) 240~310
    (x_train_r4, y_train_r4), (x_test_r4, y_test_r4) = dataloader.get_rotate_data_4()
    dataloader.save_dataset_as_batches(x_train_r4, y_train_r4, x_test_r4, y_test_r4,
                                       version_name='cifar10_rotate_4',
                                       batch_size=10000)

    # -----------------------
    # 4) Color Emphasis 4가지
    # -----------------------

    # (1) Red
    (x_train_jr, y_train_jr), (x_test_jr, y_test_jr) = dataloader.get_jitter_data_red()
    dataloader.save_dataset_as_batches(x_train_jr, y_train_jr, x_test_jr, y_test_jr,
                                       version_name='cifar10_jitter_red',
                                       batch_size=10000)

    # (2) Green
    (x_train_jg, y_train_jg), (x_test_jg, y_test_jg) = dataloader.get_jitter_data_green()
    dataloader.save_dataset_as_batches(x_train_jg, y_train_jg, x_test_jg, y_test_jg,
                                       version_name='cifar10_jitter_green',
                                       batch_size=10000)

    # (3) Orange
    (x_train_jo, y_train_jo), (x_test_jo, y_test_jo) = dataloader.get_jitter_data_orange()
    dataloader.save_dataset_as_batches(x_train_jo, y_train_jo, x_test_jo, y_test_jo,
                                       version_name='cifar10_jitter_orange',
                                       batch_size=10000)

    # (4) Purple
    (x_train_jp, y_train_jp), (x_test_jp, y_test_jp) = dataloader.get_jitter_data_purple()
    dataloader.save_dataset_as_batches(x_train_jp, y_train_jp, x_test_jp, y_test_jp,
                                       version_name='cifar10_jitter_purple',
                                       batch_size=10000)

    # -----------------------
    # 5) Frequency Filter 4가지 (Noise)
    # -----------------------
    # (1) lowpass
    (x_train_fl, y_train_fl), (x_test_fl, y_test_fl) = dataloader.get_freq_lowpass()
    dataloader.save_dataset_as_batches(x_train_fl, y_train_fl, x_test_fl, y_test_fl,
                                       version_name='cifar10_freq_lowpass',
                                       batch_size=10000)

    # (2) highpass
    (x_train_fh, y_train_fh), (x_test_fh, y_test_fh) = dataloader.get_freq_highpass()
    dataloader.save_dataset_as_batches(x_train_fh, y_train_fh, x_test_fh, y_test_fh,
                                       version_name='cifar10_freq_highpass',
                                       batch_size=10000)

    # (3) bandpass
    (x_train_fb, y_train_fb), (x_test_fb, y_test_fb) = dataloader.get_freq_bandpass()
    dataloader.save_dataset_as_batches(x_train_fb, y_train_fb, x_test_fb, y_test_fb,
                                       version_name='cifar10_freq_bandpass',
                                       batch_size=10000)

    # (4) bandstop
    (x_train_fs, y_train_fs), (x_test_fs, y_test_fs) = dataloader.get_freq_bandstop()
    dataloader.save_dataset_as_batches(x_train_fs, y_train_fs, x_test_fs, y_test_fs,
                                       version_name='cifar10_freq_bandstop',
                                       batch_size=10000)

    print("\n=== All 13 versions (Original + 12 variants) have been saved. ===")
    print("Check the folders under:", dataloader.data_dir)

    # 예시로 첫 이미지 시각화 (denormalize)
    sample_img = dataloader.denormalize(x_test[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Original Sample")
    plt.savefig('./original_test_sample')
    # plt.show()

    # -----------------------------------------------

    sample_img = dataloader.denormalize(x_test_jr[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Redish Jitter Sample")
    plt.savefig('./red_jitter_test_sample')

    sample_img = dataloader.denormalize(x_test_jg[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Greenish Jitter Sample")
    plt.savefig('./green_jitter_test_sample')

    sample_img = dataloader.denormalize(x_test_jo[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Orangish Jitter Sample")
    plt.savefig('./orange_jitter_test_sample')

    sample_img = dataloader.denormalize(x_test_jp[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Purplish Jitter Sample")
    plt.savefig('./purple_jitter_test_sample')

    # -----------------------------------------------

    sample_img = dataloader.denormalize(x_test_fl[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Low Pass Sample")
    plt.savefig('./low_pass_test_sample')

    sample_img = dataloader.denormalize(y_test_fh[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded High Pass Sample")
    plt.savefig('./high_pass_test_sample')

    sample_img = dataloader.denormalize(x_test_fb[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Band Pass Sample")
    plt.savefig('./band_pass_test_sample')

    sample_img = dataloader.denormalize(x_test_fs[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Band Stop Sample")
    plt.savefig('./band_stop_test_sample')
