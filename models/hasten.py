import json
import logging
import random
import networkx as nx
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base import BaseLearner
from utils.inc_net import hyper_net
from utils.toolkit import tensor2numpy, get_attribute
from utils.ort import compute_stable_entailment_loss
import utils.lorentz as L

random.seed(1993)
np.random.seed(1993)
num_workers = 8


class HierarchyManager(nn.Module):
    def __init__(self, device, network=None, embed_dim=512, args=None):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.network = network
        with open("knowledge/virtual_templates.json", "r") as f:
            self.virtual_templates = json.load(f)[args["dataset"]]
        self._data_to_prompt = self.virtual_templates
        self.graph = nx.DiGraph()
        self.all_class_names = []
        self.node_to_idx = {}
        self.node_list = []
        self.real_class_graph_indices_map = {}
        self.topo_order = []
        self.num_total_nodes = 0
        self.register_buffer(
            "all_prototypes", torch.zeros(0, embed_dim, device=self.device)
        )
        self.register_buffer(
            "has_proto", torch.zeros(0, dtype=torch.bool, device=self.device)
        )
        self.register_buffer(
            "mutable_nodes", torch.zeros(0, dtype=torch.bool, device=self.device)
        )
        self.root_prototypes = None

    def load_growth_hierarchy(self, hierarchy_path, current_task_class_names):
        print(f"Loading incremental hierarchy from: {hierarchy_path}")
        old_prototypes_dict = {}
        old_has_proto_dict = {}
        if self.num_total_nodes > 0:
            with torch.no_grad():
                for i, name in enumerate(self.node_list):
                    if self.has_proto[i]:
                        old_prototypes_dict[name] = self.all_prototypes[i].clone()
                        old_has_proto_dict[name] = True
        self.all_class_names = current_task_class_names
        with open(hierarchy_path, "r") as f:
            parent_to_children = json.load(f)
        self.graph = nx.DiGraph()
        all_nodes_in_hierarchy = set()
        for parent, children in parent_to_children.items():
            all_nodes_in_hierarchy.add(parent)
            for child in children:
                self.graph.add_edge(parent, child)
                all_nodes_in_hierarchy.add(child)
        self.root_nodes = [
            node for (node, in_degree) in self.graph.in_degree() if in_degree == 0
        ]
        print(f"New root nodes: {self.root_nodes}")
        self.real_nodes = set(self.all_class_names)
        self.virtual_nodes = all_nodes_in_hierarchy - self.real_nodes
        self.node_list = sorted(list(all_nodes_in_hierarchy))
        self.node_to_idx = {name: i for (i, name) in enumerate(self.node_list)}
        self.num_total_nodes = len(self.node_list)
        self.real_class_graph_indices_map = {}
        for i, name in enumerate(self.all_class_names):
            if name in self.node_to_idx:
                graph_idx = self.node_to_idx[name]
                self.real_class_graph_indices_map[i] = graph_idx
        self.topo_order = list(reversed(list(nx.topological_sort(self.graph))))
        print(f"Hierarchy updated. Total nodes: {self.num_total_nodes}")
        new_all_prototypes = torch.zeros(
            self.num_total_nodes, self.embed_dim, device=self.device
        )
        new_has_proto = torch.zeros(
            self.num_total_nodes, dtype=torch.bool, device=self.device
        )
        new_mutable_nodes = torch.zeros(
            self.num_total_nodes, dtype=torch.bool, device=self.device
        )
        restored_count = 0
        with torch.no_grad():
            for name, proto_vec in old_prototypes_dict.items():
                if name in self.node_to_idx:
                    new_idx = self.node_to_idx[name]
                    new_all_prototypes[new_idx] = proto_vec
                    new_has_proto[new_idx] = True
                    restored_count += 1
        print(f"Restored {restored_count} prototypes from previous tasks.")
        self.register_buffer("all_prototypes", new_all_prototypes)
        self.register_buffer("has_proto", new_has_proto)
        self.register_buffer("mutable_nodes", new_mutable_nodes)

    @torch.no_grad()
    def set_task_classes(self, new_real_indices: torch.Tensor):
        self.mutable_nodes.zero_()
        if new_real_indices is None or len(new_real_indices) == 0:
            return
        for rid in new_real_indices.tolist():
            if rid in self.real_class_graph_indices_map:
                gidx = self.real_class_graph_indices_map[rid]
                self.mutable_nodes[gidx] = True
        for node in self.virtual_nodes:
            gidx = self.node_to_idx[node]
            if self.has_proto[gidx]:
                self.all_prototypes[gidx] = self.all_prototypes[gidx].detach()
                self.mutable_nodes[gidx] = False

    def get_all_prototypes(
        self, real_prototypes, real_prototypes_indices, curv, iterations=3, temp=True
    ):
        out_prototypes = self.all_prototypes.clone()
        updated_graph_indices = []
        for i, real_idx in enumerate(real_prototypes_indices):
            rid = int(real_idx.item())
            if rid not in self.real_class_graph_indices_map:
                continue
            gidx = self.real_class_graph_indices_map[rid]
            if not self.mutable_nodes[gidx]:
                continue
            out_prototypes[gidx] = real_prototypes[i]
            with torch.no_grad():
                self.all_prototypes[gidx] = real_prototypes[i]
                self.has_proto[gidx] = True
            updated_graph_indices.append(gidx)
        if len(updated_graph_indices) > 0:
            affected_nodes = set()
            for gidx in updated_graph_indices:
                node_name = self.node_list[gidx]
                affected_nodes.update(nx.ancestors(self.graph, node_name))
            for _ in range(iterations):
                any_change = False
                for node_name in self.topo_order:
                    if node_name not in affected_nodes:
                        continue
                    idx = self.node_to_idx[node_name]
                    if bool(self.has_proto[idx]):
                        continue
                    children = list(self.graph.successors(node_name))
                    child_idx = [
                        self.node_to_idx[c]
                        for c in children
                        if bool(self.has_proto[self.node_to_idx[c]])
                    ]
                    if len(child_idx) == 0:
                        continue
                    with torch.no_grad():
                        templates = self._data_to_prompt
                        texts = [t.format(node_name) for t in templates]
                        texts = self.network.tokenizer(texts).cuda()
                        class_embeddings = self.network.model.encode_text(texts)
                        class_embeddings = self.network.Image_encode(class_embeddings)
                        class_embeddings = class_embeddings.mean(dim=0)
                        parent_proto = self.network.trans_hyper(class_embeddings)
                        self.all_prototypes[idx] = parent_proto
                        self.has_proto[idx] = True
                        out_prototypes[idx] = parent_proto
                        any_change = True
                if not any_change:
                    break
        return out_prototypes

    def compute_hierarchy_loss(self, all_prototypes, curv, parent_child_margin=0.2):
        losses = {
            "entailment": torch.tensor(0.0, device=self.device),
            "distance": torch.tensor(0.0, device=self.device),
            "root": torch.tensor(0.0, device=self.device),
        }
        (parent_indices_list, child_indices_list) = ([], [])
        for parent, child in self.graph.edges():
            parent_indices_list.append(self.node_to_idx[parent])
            child_indices_list.append(self.node_to_idx[child])
        if not parent_indices_list:
            return losses
        parent_indices = torch.tensor(
            parent_indices_list, device=self.device, dtype=torch.long
        )
        child_indices = torch.tensor(
            child_indices_list, device=self.device, dtype=torch.long
        )
        parent_prototypes = all_prototypes[parent_indices]
        child_prototypes = all_prototypes[child_indices]
        valid_pairs = (parent_prototypes.abs().sum(dim=-1) > 0) & (
            child_prototypes.abs().sum(dim=-1) > 0
        )
        if not valid_pairs.any():
            return losses
        parent_prototypes = parent_prototypes[valid_pairs]
        child_prototypes = child_prototypes[valid_pairs]
        parent_indices = parent_indices[valid_pairs]
        losses["entailment"] = compute_stable_entailment_loss(
            parent_prototypes, child_prototypes, curv
        )
        margin_val = torch.tensor(parent_child_margin, device=curv.device)
        marg = margin_val - L.element_dist(parent_prototypes, child_prototypes, curv)
        losses["distance"] = F.softplus(marg.mean(), beta=10) * 0.5
        root_indices = [self.node_to_idx[name] for name in self.root_nodes]
        root_prototypes = all_prototypes[root_indices]
        self.root_prototypes = root_prototypes.detach()
        valid_roots = root_prototypes.abs().sum(dim=-1) > 0
        if valid_roots.any():
            root_prototypes = root_prototypes[valid_roots]
            origin = torch.zeros(1, root_prototypes.shape[1], device=self.device)
            root_dist = L.pairwise_dist(root_prototypes, origin, curv).squeeze(-1)
            losses["root"] = torch.mean(root_dist) * 0.1
        return losses


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._train_transformer = False
        self._network = hyper_net(args)
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-08)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self._known_classes = 0
        self.prototype = []
        self.cov = []
        self.hierarchy_manager = None

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.model.encode_image(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self.hierarchy_manager is None:
            all_class_names = data_manager._class_to_label
            self.class_len = len(all_class_names)
            self.hierarchy_manager = HierarchyManager(
                self._device, self._network, args=self.args
            )
        hierarchy_path = f"{self.args['hierarchy_dir']}/task_{self._cur_task}.json"
        self.hierarchy_manager.load_growth_hierarchy(
            hierarchy_path, data_manager._class_to_label[: self._total_classes]
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.update_task()
        new_class_indices = torch.arange(
            self._known_classes, self._total_classes, device=self._device
        )
        self.hierarchy_manager.set_task_classes(new_class_indices)
        self.train(self.train_loader, self.test_loader, train_dataset)
        self._network.compute_cov(
            self._known_classes,
            self._total_classes,
            self.train_loader,
            self._device,
            data_manager,
        )

    def train(self, train_loader, test_loader, train_dataset):
        self._network.to(self._device)
        self.replace_fc(train_loader, self._network, None)
        trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                trainable_params,
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.AdamW(
                trainable_params, lr=self.init_lr, weight_decay=self.weight_decay
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
        )
        class_to_label = self.data_manager._class_to_label
        prog_bar = tqdm(range(self.tuned_epoch))
        total_labels = class_to_label[: self._total_classes]
        templates = self.data_manager._data_to_prompt[0]
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            (correct, total) = (0, 0)
            correct_clip = []
            if self._cur_task > 0:
                old_class = list(
                    range(
                        self.args["init_cls"]
                        + (self._cur_task - 1) * self.args["increment"]
                    )
                )
                random.shuffle(old_class)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                original_batch_size = inputs.shape[0]
                if self._cur_task > 0:
                    proto_x = []
                    proto_y = []
                    old_class = (
                        random.choices(old_class, k=self.args["sample_num"])
                        if self.args["sample_num"] > 0
                        else old_class
                    )
                    for j in old_class:
                        num_samples = 1
                        dist = torch.distributions.MultivariateNormal(
                            self.prototype[j], covariance_matrix=self.cov[j]
                        )
                        proto_x.append(dist.sample((num_samples,)))
                        proto_y.append(
                            torch.full(
                                (num_samples,), j, dtype=torch.long, device=self._device
                            )
                        )
                    proto_x = torch.cat(proto_x, dim=0)
                    proto_y = torch.cat(proto_y, dim=0)
                    targets = torch.cat([targets, proto_y], dim=0)
                _curv = self._network.get_curv()
                _scale = self._network.logit_scale.exp()
                labels = [class_to_label[y] for y in targets]
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features
                image_features = self._network.encode_image(inputs)
                img_feas = image_features
                if self._cur_task > 0:
                    sg_image_features = self._network.Image_encode(proto_x)
                    sg_image_features = self._network.trans_hyper(sg_image_features)
                    img_feas = torch.cat([img_feas, sg_image_features], dim=0)
                logits = -L.pairwise_dist(img_feas, text_feas, _curv)
                texts_clip = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(
                    self._network.tokenizer(texts_clip).to(self._device)
                )
                image_features_hyperbolic = img_feas
                text_features_hyperbolic = clip_text_feas
                all_image_feats = image_features_hyperbolic
                all_text_feats = text_features_hyperbolic
                image_logits = -L.pairwise_dist(all_image_feats, all_text_feats, _curv)
                text_logits = image_logits.T
                loss_targets = torch.arange(
                    all_image_feats.shape[0], device=self._device
                )
                contrastive_loss = (
                    F.cross_entropy(_scale * image_logits, loss_targets)
                    + F.cross_entropy(_scale * text_logits, loss_targets)
                ) * 0.5
                entailment_loss = torch.tensor(0.0, device=self._device)
                if self.args["entail_weight"] > 0 and original_batch_size > 0:
                    real_image_feats = image_features_hyperbolic
                    real_text_feats = text_features_hyperbolic
                    entailment_loss = compute_stable_entailment_loss(
                        real_text_feats,
                        real_image_feats,
                        _curv,
                        min_radius=self.args["kcone"],
                    )
                all_real_prototypes_seen = text_feas
                seen_class_indices = torch.arange(
                    self._total_classes, device=self._device
                )
                all_node_prototypes = self.hierarchy_manager.get_all_prototypes(
                    all_real_prototypes_seen, seen_class_indices, _curv
                )
                hierarchy_loss = self.hierarchy_manager.compute_hierarchy_loss(
                    all_node_prototypes, _curv, parent_child_margin=self.args["margin"]
                )
                loss = (
                    contrastive_loss
                    + self.args["entail_weight"] * entailment_loss
                    + self.args["hierarchy_weight"]
                    * (
                        hierarchy_loss["entailment"]
                        + hierarchy_loss["distance"]
                        + hierarchy_loss["root"]
                    )
                )
                if epoch < int(self.tuned_epoch) and self.args["mapping_mode"] != "fix":
                    with torch.no_grad():
                        target_hyp = self._network.model.encode_image(inputs)
                        target_hyp = self._network.Image_encode(target_hyp)
                        if self._cur_task > 0:
                            proto_x_a = self._network.Image_encode(proto_x)
                            target_hyp = torch.cat([target_hyp, proto_x_a], dim=0)
                        target_hyp = L.exp_map0(
                            target_hyp, curv=self._network.get_curv()
                        )
                    pred_hyp = image_features_hyperbolic
                    lambda_geom = self.args["reg_weight"]
                    map_reg = (
                        lambda_geom
                        * (pred_hyp - target_hyp.detach()).pow(2).sum(dim=-1).mean()
                    )
                    loss += map_reg
                optimizer.zero_grad()
                loss.backward()
                if self._cur_task > 0 and self.args["project_grad"]:
                    with torch.no_grad():
                        if self._network.learned_lorentz_exp:
                            lle = self._network.learned_lorentz_exp.exp_map[0]
                            projected_grad = (
                                self._network.learned_lorentz_exp.project_gradient(
                                    lle.weight.grad
                                )
                            )
                            lle.weight.grad = projected_grad
                optimizer.step()
                losses += loss.item()
                (_, preds) = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                correct_clip.append(preds.eq(targets).cpu())
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader, epoch)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        self._network.eval()
        sample_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False, num_workers=8
        )
        x = []
        y = []
        for i, (_, inputs, targets) in enumerate(sample_loader):
            with torch.no_grad():
                (inputs, targets) = (inputs.to(self._device), targets.to(self._device))
                fea = self._network.model.encode_image(inputs)
                x.append(fea)
                y.append(targets)
        y = torch.cat(y, dim=0)
        x = torch.cat(x, dim=0)
        label = torch.sort(torch.unique(y))[0]
        for l in label:
            index = torch.nonzero(y == l).squeeze()
            l_fea = x[index]
            l_mean = l_fea.mean(dim=0)
            self.prototype.append(l_mean)
            l_cov = torch.cov((l_fea - l_mean).T) + 1e-05 * torch.eye(
                self._network.feature_dim, device=self._device
            )
            self.cov.append(l_cov)
        self._network.eval()

    def _compute_accuracy(self, model, loader, epoch=0):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[: self._total_classes]
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.model.encode_text(texts)
                class_embeddings = self._network.Text_encode(class_embeddings)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        (correct, total) = (0, 0)
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features = self._network.model.encode_image(inputs)
                transf_image_features = self._network.Image_encode(
                    transf_image_features
                )
                transf_text_features = text_features
                if epoch == self.args["tuned_epoch"] - 1:
                    transf_image_features_raw_ = self._network.visual_forward_(inputs)
                    transf_image_features_raw = (
                        transf_image_features_raw_ @ self._network.visual_proj
                    )
                    transf_image_features_raw_ = (
                        transf_image_features_raw_
                        / transf_image_features_raw_.norm(dim=-1, keepdim=True)
                    )
                    transf_image_features_raw = (
                        transf_image_features_raw
                        / transf_image_features_raw.norm(dim=-1, keepdim=True)
                    )
                    transf_image_features = self._network.trans_hyper(
                        transf_image_features
                    )
                    transf_text_features = self._network.trans_hyper(
                        transf_text_features
                    )
                    outputs = -L.pairwise_dist(
                        transf_image_features,
                        transf_text_features,
                        self._network.get_curv(),
                    )
                    outputs_sp = self._network.fc(transf_image_features_raw)["logits"]
                    outputs = self.get_ens(outputs, outputs_sp)
                else:
                    transf_image_features = self._network.trans_hyper(
                        transf_image_features
                    )
                    transf_text_features = self._network.trans_hyper(
                        transf_text_features
                    )
                    outputs = -L.pairwise_dist(
                        transf_image_features,
                        transf_text_features,
                        self._network.get_curv(),
                    )
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        templates = templates[:1]
        total_labels = class_to_label[: self._total_classes]
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.model.encode_text(texts)
                class_embeddings = self._network.Text_encode(class_embeddings)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
            text_features_hyp = self._network.trans_hyper(text_features)
        text_features = text_features_hyp.to(self._device)
        transf_text_features = text_features_hyp
        y_true = []
        y_esp = []
        for _, (idx, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features = self._network.model.encode_image(inputs)
                transf_image_features = self._network.Image_encode(
                    transf_image_features
                )
                transf_image_features = self._network.trans_hyper(transf_image_features)
                transf_image_features_raw_ = self._network.visual_forward_(inputs)
                transf_image_features_raw = (
                    transf_image_features_raw_ @ self._network.visual_proj
                )
                transf_image_features_raw_ = (
                    transf_image_features_raw_
                    / transf_image_features_raw_.norm(dim=-1, keepdim=True)
                )
                transf_image_features_raw = (
                    transf_image_features_raw
                    / transf_image_features_raw.norm(dim=-1, keepdim=True)
                )
                outputs = -L.pairwise_dist(
                    transf_image_features,
                    transf_text_features,
                    self._network.get_curv(),
                )
                outputs_sp = self._network.fc(transf_image_features_raw)["logits"]
                outputs_esp = self.get_ens(outputs, outputs_sp)
            esp_predicts = torch.topk(
                outputs_esp, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_true.append(targets.cpu().numpy())
            y_esp.append(esp_predicts.cpu().numpy())
        return ([np.concatenate(y_esp)], np.concatenate(y_true))

    def get_ens(self, outputs, outputs_complement):
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        outputs_complement = outputs_complement / outputs_complement.norm(
            dim=-1, keepdim=True
        )
        outputs = F.softmax(outputs, dim=-1)
        outputs_complement = F.softmax(outputs_complement, dim=-1)
        return outputs * self.args["stat"] + outputs_complement * (
            1 - self.args["stat"]
        )
