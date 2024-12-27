import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_prepare.data_preprocess import prepareData_accWithSTFT_2d


def select_dataset(dataset_name):
    dataloader = None

    if dataset_name == 'cifar-10':
        from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
        dataloader = cifar10Dataloader('./dataset/cifar10')
    elif dataset_name == 'cifar-100':
        from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
        dataloader = cifar100Dataloader('./dataset/cifar100')
    elif dataset_name == 'mnist':
        from dataset.mnist.mnistDataLoader import mnistDataloader
        dataloader = mnistDataloader(data_dir='./dataset/mnist/')
    elif dataset_name == 'svhn':
        from dataset.svhn.svhn_dataloader import svhnDataloader
        dataloader = svhnDataloader(data_dir='./dataset/svhn/')

    (x_train, y_train), (x_test, y_test) = dataloader.load_data()
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
