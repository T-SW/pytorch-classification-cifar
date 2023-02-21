import os
import cv2

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from conf import settings


class MyDataset(Dataset):
    def __init__(self, data_dir, label_path, transform=None):
        super(MyDataset, self).__init__()

        self.data_list = []
        with open(label_path, encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path)
        image = image.astype('float32')
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        return image, label

    def __len__(self):
        return len(self.data_list)


data_path = r"./data"


def get_train_dataloader(package, data_dir, batch_size, num_workers, shuffle=True, mean=settings.MEAN_10, std=settings.STD_10):
    if package == 'CIFAR100':
        mean = settings.MEAN_100
        std = settings.STD_100
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding=4),                           # 随机中心裁剪
        transforms.RandomHorizontalFlip(),                              # 随机水平镜像
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if package == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True)
    elif package == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=transform_train, download=True)
    else:
        train_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    return train_loader


def get_test_dataloader(data_dir, batch_size, num_workers, package=False, shuffle=True, mean=settings.MEAN_10, std=settings.STD_10):
    if package == 'CIFAR100':
        mean = settings.MEAN_100
        std = settings.STD_100
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if package == 'CIFAR10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform_test, download=True)
    elif package == 'CIFAR100':
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, transform=transform_test, download=True)
    else:
        test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_test)
    print(test_dataset.class_to_idx)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers)

    return test_loader
