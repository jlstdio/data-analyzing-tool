import numpy as np
import os
import urllib.request
from scipy.io import loadmat

class svhnDataloader(object):

    '''
    # 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
    -> 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data

    # Comes in two formats:
        1. Original images with character level bounding boxes.
        2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

    this one chose MNIST like 32x32 images
    '''

    TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

    TRAIN_FILE = 'train_32x32.mat'
    TEST_FILE = 'test_32x32.mat'

    def __init__(self, data_dir, normalize=True):
        """
        SVHN 데이터 로더 초기화.

        Args:
            data_dir (str): SVHN 데이터 파일이 위치한 디렉토리 경로.
            normalize (bool): 데이터를 정규화할지 여부. 기본값은 True.
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.mean = None
        self.std = None

        # Ensure the data directory exists
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}")

        # Check if all required files are present; if not, download them
        if not self._check_files_exist():
            print("Required SVHN files not found. Downloading dataset...")
            self._download_file(self.TRAIN_URL, self.TRAIN_FILE)
            self._download_file(self.TEST_URL, self.TEST_FILE)
            print("Download complete.")
        else:
            print("All SVHN files are present.")

    def _check_files_exist(self):
        """
        Check if the SVHN train and test files exist in the data directory.

        Returns:
            bool: True if both files exist, False otherwise.
        """
        train_path = os.path.join(self.data_dir, self.TRAIN_FILE)
        test_path = os.path.join(self.data_dir, self.TEST_FILE)
        for file_path in [train_path, test_path]:
            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                return False
        return True

    def _download_file(self, url, filename):
        """
        Download a file from a given URL into the data directory.
        """
        file_path = os.path.join(self.data_dir, filename)
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, file_path, self._download_progress)
            print(f"\n{filename} downloaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename}: {e}")

    def _download_progress(self, block_num, block_size, total_size):
        """
        Display download progress.

        Args:
            block_num (int): Number of blocks transferred so far.
            block_size (int): Block size in bytes.
            total_size (int): Total size of the file.
        """
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100
        percent = min(100, percent)
        print(f"\rDownload progress: {percent:.2f}%", end='')

    def _load_mat_file(self, filename):
        """
        Load a .mat file and extract images and labels.

        Returns:
            (data, labels): data는 shape가 (N, 32, 32, 3)인 numpy 배열,
                            labels는 shape가 (N,)인 numpy 배열.
        """
        file_path = os.path.join(self.data_dir, filename)
        mat = loadmat(file_path)
        # SVHN format: mat['X']는 (32,32,3,N) 형태이고 mat['y']는 (N,1) 형태
        data = mat['X']
        labels = mat['y'].flatten()

        # Transpose data to (N,32,32,3)
        data = np.transpose(data, (3, 0, 1, 2))

        # SVHN의 라벨은 1~10 범위를 가지며, 10은 실제 0을 의미
        # 0~9 범위로 맞추고 싶다면 아래와 같이 변환
        labels = (labels % 10)

        return data, labels

    def load_data(self):
        x_train, y_train = self._load_mat_file(self.TRAIN_FILE)
        x_test, y_test = self._load_mat_file(self.TEST_FILE)

        # 정규화 수행
        if self.normalize:
            self.mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
            self.std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
            print(f"Computed mean: {self.mean.flatten()}")
            print(f"Computed std: {self.std.flatten()}")

            # 정규화 적용
            x_train = (x_train - self.mean) / self.std
            x_test = (x_test - self.mean) / self.std

        return (x_train, y_train), (x_test, y_test)

    def get_mean_std(self):
        """
        계산된 평균과 표준편차를 반환.

        Returns:
            tuple: (mean, std) 각각은 (1, 1, 1, 3) 형태의 numpy 배열.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been computed. Please call load_data() first.")
        return self.mean, self.std
