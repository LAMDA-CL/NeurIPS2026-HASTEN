import numpy as np
from utils.toolkit import accuracy


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        (self._data_memory, self._targets_memory) = (np.array([]), np.array([]))
        self.topk = 5
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def _evaluate(self, y_pred, y_true):
        ret = {}
        if len(y_pred.shape) == 1:
            grouped = accuracy(y_pred, y_true, self._known_classes)
            ret["grouped"] = grouped
            ret["top1"] = grouped["total"]
        else:
            grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
            ret["grouped"] = grouped
            ret["top1"] = grouped["total"]
            ret["top{}".format(self.topk)] = np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
                decimals=2,
            )
        return ret

    def eval_task(self):
        (y_pred, y_true) = self._eval_cnn(self.test_loader)
        cnn_accy = [self._evaluate(y_pred[0], y_true)]
        return (cnn_accy, None, None, None, None, None)
