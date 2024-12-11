import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import sys




import torch
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(pil_loader(self.images[idx]))
        label = self.labels[idx]

        return image, label
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        # print(f"test_dataset path is{self.test_data} ,and {test_dataset.root}")


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet100/train.txt"
        test_dir = "./data/imagenet100/eval.txt"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet100/train"
        test_dir = "./data/imagenet100/eval"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class tinyImageNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    test_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_trsf = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        test_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        #assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/tiny-imagenet-200/train"
        test_dir = "./data/tiny-imagenet-200/val"
        #
        # train_dset = datasets.ImageFolder(train_dir)
        # self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        #
        #
        #
        #
        #
        # test_dset = datasets.ImageFolder(test_dir)
        #
        #
        # self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        train_dset = datasets.ImageFolder(train_dir)

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
        self.train_data, self.train_targets = np.array(train_images), np.array(train_labels)

        test_images = []
        test_labels = []

        _, class_to_idx = find_classes(train_dir)
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        self.test_data, self.test_targets = np.array(test_images), np.array(test_labels)

        # change order

        # order = [i for i in range(len(np.unique(train_targets)))]
        # np.random.seed(seed)
        # order = np.random.permutation(len(order)).tolist()
        #
        # logging.info(order)
        #
        # train_targets = _map_new_class_index(train_targets, order)
        # test_targets = _map_new_class_index(test_targets, order)

        # trainfolder = DummyDataset(train_data, train_targets, train_trsf)
        # testfolder = DummyDataset(test_data, test_targets, test_trsf)
        #
        # train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=128,
        #                                            shuffle=True, drop_last=True, num_workers=8)
        #
        # val_loader = torch.utils.data.DataLoader(testfolder, batch_size=128,
        #                                          shuffle=False, drop_last=False, num_workers=8)
        #
        # self.train_data, self.train_targets = split_images_labels(train_loader.imgs)
        # self.test_data, self.test_targets = split_images_labels(val_loader.imgs)





class iLibrispeech100(iData):
    use_path = False

    class_order = np.arange(100).tolist()

    def download_data(self):
        
        train_dataset = LBRS(root="./data/ls100/librispeech/train-clean-100-sgments/", phase="train")
        test_dataset = LBRS(root="./data/ls100/librispeech/train-clean-100-sgments/", phase="test")

        self.train_data, self.train_targets = train_dataset.data, np.array( 
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )