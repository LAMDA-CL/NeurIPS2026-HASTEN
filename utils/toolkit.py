import numpy as np


def count_parameters(model, trainable=False):
    if trainable:
        return sum((p.numel() for p in model.parameters() if p.requires_grad))
    return sum((p.numel() for p in model.parameters()))


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )
    all_acc["harmonic"] = np.around(
        2 * all_acc["old"] * all_acc["new"] / (all_acc["old"] + all_acc["new"]),
        decimals=2,
    )
    return all_acc


def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
    return (np.array(images), np.array(labels))


def get_attribute(dic, name, default):
    if name in dic:
        return dic[name]
    else:
        print(name, "not in args, set to", default, " as default")
        return default
