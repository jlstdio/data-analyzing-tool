import os
import shutil
import tarfile
import urllib.request
import pickle

import numpy as np
from scipy.ndimage import rotate  # random rotate

import matplotlib
import matplotlib.pyplot as plt


class cifar10_expanded_dataloader(object):
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

    # ---------------------- Augmentation 함수들 ----------------------
    def _color_jitter(self, images, jitter_scale=0.6, jitter_offset=0.6):
        """
        채널별로 랜덤 밝기/스케일링을 가미하여 색감 변화
        images shape: (N, 32, 32, 3)
        """
        images_jittered = images.copy()
        N = images_jittered.shape[0]

        for i in range(N):
            for c in range(3):
                offset = np.random.uniform(-jitter_offset, jitter_offset)
                scale = 1.0 + np.random.uniform(-jitter_scale, jitter_scale)
                images_jittered[i, :, :, c] = (images_jittered[i, :, :, c] + offset) * scale

        return images_jittered

    def _random_rotate(self, images):
        """
        images shape: (N, 32, 32, 3)
        """
        possible_angles = np.arange(-10, 10, 1)
        images_rotated = np.zeros_like(images)
        N = images.shape[0]
        for i in range(N):
            angle = np.random.choice(possible_angles)
            rotated_img = rotate(images[i], angle, reshape=False, axes=(0, 1))
            images_rotated[i] = rotated_img
        return images_rotated

    def _add_noise(self, images, noise_level=0.3):
        """
        가우시안 노이즈 추가
        images shape: (N, 32, 32, 3)
        noise_level: 노이즈 강도 (표준편차)
        """
        noisy_images = images.copy()
        noise = np.random.randn(*images.shape) * noise_level
        noisy_images += noise
        return noisy_images

    # ---------------------- 4가지 버전 데이터셋 반환 함수 ----------------------
    def get_original_data(self):
        """원본 데이터셋 (정규화된 상태)"""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Please call load_data() first.")
        return (self.x_train.copy(), self.y_train.copy()), (self.x_test.copy(), self.y_test.copy())

    def get_jitter_data(self):
        """color jitter 버전"""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Please call load_data() first.")

        x_train_jitter = self._color_jitter(self.x_train)
        x_test_jitter = self._color_jitter(self.x_test)

        return (x_train_jitter, self.y_train.copy()), (x_test_jitter, self.y_test.copy())

    def get_rotate_data(self):
        """random rotate 버전"""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Please call load_data() first.")

        x_train_rotate = self._random_rotate(self.x_train)
        x_test_rotate = self._random_rotate(self.x_test)

        return (x_train_rotate, self.y_train.copy()), (x_test_rotate, self.y_test.copy())

    def get_noise_data(self):
        """noise 추가 버전"""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Please call load_data() first.")

        x_train_noise = self._add_noise(self.x_train)
        x_test_noise = self._add_noise(self.x_test)

        return (x_train_noise, self.y_train.copy()), (x_test_noise, self.y_test.copy())

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
            ./[data_dir]/cifar10_original/test_batch_0.pkl  (or test_batch.pkl)
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

        # train_batch_0.pkl, train_batch_1.pkl ... 순서대로 로드
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
    # 1) 데이터 로드 및 4가지 버전 생성
    dataloader = cifar10_expanded_dataloader('./cifar10_expanded')

    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    # 2) 네 가지 버전 데이터셋 얻기
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = dataloader.get_original_data()
    (x_train_jit, y_train_jit), (x_test_jit, y_test_jit) = dataloader.get_jitter_data()
    (x_train_rot, y_train_rot), (x_test_rot, y_test_rot) = dataloader.get_rotate_data()
    (x_train_noise, y_train_noise), (x_test_noise, y_test_noise) = dataloader.get_noise_data()

    # 3) 각 버전별로 batch 파일로 저장
    #    예) cifar10_original, cifar10_jitter, cifar10_rotate, cifar10_noise
    dataloader.save_dataset_as_batches(x_train_orig, y_train_orig, x_test_orig, y_test_orig, version_name='cifar10_original', batch_size=10000)

    dataloader.save_dataset_as_batches(x_train_jit, y_train_jit, x_test_jit, y_test_jit, version_name='cifar10_jitter', batch_size=10000)

    dataloader.save_dataset_as_batches(x_train_rot, y_train_rot, x_test_rot, y_test_rot, version_name='cifar10_rotate', batch_size=10000)

    dataloader.save_dataset_as_batches(x_train_noise, y_train_noise, x_test_noise, y_test_noise, version_name='cifar10_noise', batch_size=10000)

    print("\n--- All datasets saved as batch files. ---\n")

    '''
    # 4) 필요할 때, 저장된 batch 파일에서 즉시 로드하여 사용 가능
    (x_train_orig_load, y_train_orig_load), (x_test_orig_load, y_test_orig_load) = dataloader.load_dataset_from_batches("cifar10_original")
    print("Loaded original dataset from batch files:", x_train_orig_load.shape, x_test_orig_load.shape)

    # 예시로 첫 이미지 시각화 (denormalize)
    sample_img = dataloader.denormalize(x_train_orig_load[0])
    sample_img = np.squeeze(sample_img, axis=0)
    plt.imshow(sample_img)
    plt.title("Loaded Original Sample")
    plt.show()
    '''