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


class Data_enhancement():
    """
    transforms数据增强整理：  https://zhuanlan.zhihu.com/p/166130922
    """
    def __init__(self, mean, std, img_size):
        self.mean = mean
        self.std = std
        self.img_size = img_size

    def cifar(self):
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
            # transforms.RandomRotation(15),
            transforms.RandomCrop(30),                           # 随机中心裁剪
            transforms.RandomHorizontalFlip(),                              # 随机水平镜像
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transform_train

    def General(self):
        transform_train = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomVerticalFlip(p=0.3),                           # 随机垂直镜像
            transforms.RandomHorizontalFlip(),                              # 随机水平镜像
            # transforms.RandomErasing(scale=(0.02, 0.04), ratio=(0.5, 2)),   # 随机遮挡
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transform_train


def get_train_dataloader(package, data_dir, img_size, mean, std, batch_size, num_workers, shuffle=True):
    if not mean:
        mean = settings.TRAIN_MEAN
    if not std:
        std = settings.TRAIN_STD
    transform_train = Data_enhancement(mean, std, img_size)

    if package:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform_train.cifar(), download=True)
    else:
        train_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_train.General())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    return train_loader


def get_test_dataloader(package, data_dir, img_size, mean, std, batch_size, num_workers, shuffle=True):
    if not mean:
        mean = settings.TRAIN_MEAN
    if not std:
        std = settings.TRAIN_STD
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if package:
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=True)
    else:
        test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers)

    return test_loader
