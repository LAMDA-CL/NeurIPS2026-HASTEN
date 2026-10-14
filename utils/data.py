import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

DATA_ROOT_PATH = "data"


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args):
    t = [
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return t


class iCIFAR224(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        data_path = os.path.join(DATA_ROOT_PATH, "cifar100")
        train_dataset = datasets.cifar.CIFAR100(data_path, train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100(data_path, train=False, download=False)
        (self.train_data, self.train_targets) = (
            train_dataset.data,
            np.array(train_dataset.targets),
        )
        (self.test_data, self.test_targets) = (
            test_dataset.data,
            np.array(test_dataset.targets),
        )


class iImageNetR(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "imagenet-r/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "imagenet-r/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "objectnet/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "objectnet/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "cub/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "cub/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class Food101(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "food101/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "food101/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class Aircraft(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "aircraft/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "aircraft/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class UCF101(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "ucf/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "ucf/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class StanfordCars(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "cars/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "cars/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)


class SUN(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(300).tolist()

    def download_data(self):
        train_dir = os.path.join(DATA_ROOT_PATH, "sun/train/")
        test_dir = os.path.join(DATA_ROOT_PATH, "sun/test/")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        print(train_dset.class_to_idx)
        (self.train_data, self.train_targets) = split_images_labels(train_dset.imgs)
        (self.test_data, self.test_targets) = split_images_labels(test_dset.imgs)
