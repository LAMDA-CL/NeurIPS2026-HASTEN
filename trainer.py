import copy
import logging
import os
import random
import sys
import numpy as np
import torch
from models.hasten import Learner
from utils.data_manager import DataManager
from utils.create_hier import prepare_hierarchies
from utils.toolkit import count_parameters

np.set_printoptions(precision=4, suppress=True, formatter={"float": "{: 0.4f}".format})


def train(args):
    device = copy.deepcopy(args["device"])
    args["device"] = device
    _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}/{}/{}/seed:{}_{}_model:{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["seed"],
        args["project_name"],
        args["model_name"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    args["hierarchy_dir"] = str(prepare_hierarchies(data_manager, args))
    model = Learner(args)
    model.save_dir = logs_name
    cnn_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        (cnn_accy, _, _, _, _, _) = model.eval_task()
        model.after_task()
        if len(cnn_accy) == 1:
            cnn_accy = cnn_accy[0]
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
            print(
                "Average Accuracy (CNN):",
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (CNN): {}".format(
                    sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                )
            )


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
