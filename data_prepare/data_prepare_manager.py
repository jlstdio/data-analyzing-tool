import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_prepare.data_preprocess import prepareData_accWithSTFT_2d


def select_dataset(dataset_name):
    dataloader = None
    (x_train, y_train), (x_test, y_test) = (None, None), (None, None)

    if dataset_name == 'cifar-10':
        from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
        dataloader = cifar10Dataloader('./dataset/cifar10')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'cifar-10_jitter':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.load_dataset_from_batches("cifar10_jitter")
    elif dataset_name == 'cifar-10_rotate':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.load_dataset_from_batches("cifar10_rotate")
    elif dataset_name == 'cifar-10_noise':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.load_dataset_from_batches("cifar10_noise")
    elif dataset_name == 'cifar-100':
        from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
        dataloader = cifar100Dataloader('./dataset/cifar100')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'mnist':
        from dataset.mnist.mnistDataLoader import mnistDataloader
        dataloader = mnistDataloader(data_dir='./dataset/mnist/')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'svhn':
        from dataset.svhn.svhn_dataloader import svhnDataloader
        dataloader = svhnDataloader(data_dir='./dataset/svhn/')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'cifar-10_jg':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_green()
    elif dataset_name == 'cifar-10_jo':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_orange()
    elif dataset_name == 'cifar-10_jr':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_red()
    elif dataset_name == 'cifar-10_jp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_purple()
    elif dataset_name == 'cifar-10_r1':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_1()
    elif dataset_name == 'cifar-10_r2':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_2()
    elif dataset_name == 'cifar-10_r3':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_3()
    elif dataset_name == 'cifar-10_r4':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_4()
    elif dataset_name == 'cifar-10_lp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_lowpass()
    elif dataset_name == 'cifar-10_bp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_bandpass()
    elif dataset_name == 'cifar-10_bs':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_bandstop()

    classes = list(set(y_test))

    # trainDataset = zip(y_train, x_train)
    testDataset = zip(y_test, x_test)

    validation_y, validation_x = zip(*testDataset)

    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)

    X_validation = torch.tensor(validation_x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_validation = torch.tensor(validation_y, dtype=torch.long)

    validation_dataset = TensorDataset(X_validation, y_validation)

    val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    return val_loader, classes


def select_custom_dataset(dataset_name, preprocess_mode):
    if preprocess_mode == "prepareData_accWithSTFT_2d":
        x_test, y_test = prepareData_accWithSTFT_2d('./dataset/data', dataset_name, 32, mode='train', train_ratio=0.9)

    classes = list(set(y_test))

    testDataset = zip(y_test, x_test)

    validation_y, validation_x = zip(*testDataset)

    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)

    # [32, 4, 3, 3], expected input[32, 301, 4, 22] 0 1 2 3 -> 0 2 3 1
    X_validation = torch.tensor(validation_x, dtype=torch.float32)#.permute(0, 2, 3, 1)
    y_validation = torch.tensor(validation_y, dtype=torch.long)

    validation_dataset = TensorDataset(X_validation, y_validation)
    val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    return val_loader, classes
