import json
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import (
    iCIFAR224,
    iImageNetR,
    objectnet,
    CUB,
    Food101,
    Aircraft,
    UCF101,
    StanfordCars,
    SUN,
)


class DataManager(object):
    def __init__(
        self,
        dataset_name,
        shuffle,
        seed,
        init_cls,
        increment,
        train_trans=None,
        test_trans=None,
    ):
        with open("./utils/labels.json", "r") as f:
            self._class_to_label = json.load(f)[dataset_name]
        print(self._class_to_label)
        with open("./utils/templates.json", "r") as f:
            self._data_to_prompt = json.load(f)[dataset_name]
        print(self._data_to_prompt)
        if train_trans is None:
            self.new_trans = False
        else:
            self.new_trans = True
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed, train_trans, test_trans)
        if init_cls > len(self._class_order):
            print("No enough classes.")
            self._increments = [len(self._class_order)]
        else:
            self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        print("Training class stages:", self._increments)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            (x, y) = (self._train_data, self._train_targets)
        elif source == "test":
            (x, y) = (self._test_data, self._test_targets)
        else:
            raise ValueError("Unknown data source {}.".format(source))
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))
        (data, targets) = ([], [])
        for idx in indices:
            if m_rate is None:
                (class_data, class_targets) = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                (class_data, class_targets) = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
        if appendent is not None and len(appendent) != 0:
            (appendent_data, appendent_targets) = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        (data, targets) = (np.concatenate(data), np.concatenate(targets))
        if ret_data:
            return (data, targets, DummyDataset(data, targets, trsf, self.use_path))
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, train_trans, test_trans):
        idata = _get_idata(dataset_name)
        idata.download_data()
        (self._train_data, self._train_targets) = (
            idata.train_data,
            idata.train_targets,
        )
        (self._test_data, self._test_targets) = (idata.test_data, idata.test_targets)
        self.use_path = idata.use_path
        if self.new_trans:
            self._train_trsf = train_trans
            self._test_trsf = test_trans
        else:
            self._train_trsf = idata.train_trsf
            self._test_trsf = idata.test_trsf
            self._common_trsf = idata.common_trsf
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        _class_to_label = [self._class_to_label[i] for i in self._class_order]
        self._class_to_label = _class_to_label
        print("After shuffle, class_to_label is: ", self._class_to_label)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return (x[idxes], y[idxes])

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return (x[new_idxes], y[new_idxes])


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return (idx, image, label)


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar224":
        return iCIFAR224()
    elif name == "imagenetr":
        return iImageNetR()
    elif name == "objectnet":
        return objectnet()
    elif name == "cub":
        return CUB()
    elif name == "food101":
        return Food101()
    elif name == "aircraft":
        return Aircraft()
    elif name == "ucf101":
        return UCF101()
    elif name == "cars":
        return StanfordCars()
    elif name == "sun":
        return SUN()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
