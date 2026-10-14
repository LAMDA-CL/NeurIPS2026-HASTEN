import copy
import torch
from torch import nn
from convs.linears import CosineLinear


def get_convnet(args, pretrained=False):
    backbone_name = args["convnet_type"].lower()
    if "clip" in backbone_name:
        print("Using CLIP model as the backbone")
        import open_clip

        if backbone_name == "clip":
            (model, _, preprocess) = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="laion400m_e32", cache_dir="cache_dir"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16")
            model.out_dim = 512
            return (model, preprocess, tokenizer)
        elif backbone_name == "clip_laion2b":
            (model, _, preprocess) = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="laion2b_s34b_b88k", cache_dir="cache_dir"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16")
            model.out_dim = 512
            return (model, preprocess, tokenizer)
        elif backbone_name == "openai_clip":
            (model, _, preprocess) = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai", cache_dir="cache_dir"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16")
            model.out_dim = 512
            return (model, preprocess, tokenizer)
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None


class AMLP_Adapter(nn.Module):
    def __init__(self, c_in, hidden, init_alpha=512 ** (-0.5)):
        super(AMLP_Adapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, hidden))
        self.log_alpha = nn.Parameter(torch.tensor(init_alpha).log())

    def forward(self, x):
        x_ = self.fc(x)
        self.log_alpha.data = self.log_alpha.data.clamp(max=0)
        x_ = x_ * self.log_alpha.exp()
        return x_


class LearnableLorentzExp(nn.Module):
    """
    Learnable approximation to Lorentz exponential map exp_map0.
    """

    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.exp_map = nn.Sequential(nn.Linear(hidden_dim, dim, bias=False))
        self.uncentered_cov = torch.zeros(dim, dim)
        self.null_space_basis = None
        self.n_total_samples = 0

    def forward(self, x: torch.Tensor):
        x = self.exp_map(x)
        return x

    def compute_null_space(self, energy_threshold=0.95):
        C = self.accumulated_cov
        orig_dtype = C.dtype
        min_keep = C.shape[0] - 10
        (U, Sval, Vh) = torch.svd(C.to(torch.float64))
        total_energy = (Sval**2).sum().item()
        accumulated_energy = 0.0
        r = 0
        for i in range(len(Sval)):
            accumulated_energy += Sval[i] ** 2
            if accumulated_energy / total_energy >= energy_threshold:
                r = i + 1
                break
        r = min(r, min_keep)
        V = Vh.T
        self.null_space_basis = V[:, r:].to(dtype=orig_dtype).detach()

    def project_gradient(self, gradient):
        if self.null_space_basis is None:
            return gradient
        basis = self.null_space_basis.to(gradient.device, dtype=gradient.dtype)
        trans = basis @ basis.T
        projected = trans @ gradient
        return projected

    def update_covariance(self, task_covariance, n_task_samples):
        if self.n_total_samples == 0:
            self.accumulated_cov = task_covariance.clone()
        else:
            old_weight = self.n_total_samples / (self.n_total_samples + n_task_samples)
            new_weight = n_task_samples / (self.n_total_samples + n_task_samples)
            self.accumulated_cov = (
                old_weight * self.accumulated_cov + new_weight * task_covariance
            )
        self.n_total_samples += n_task_samples
        self.accumulated_cov = torch.nan_to_num(
            self.accumulated_cov, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.accumulated_cov += 1e-06 * torch.eye(
            self.accumulated_cov.size(0), device=self.accumulated_cov.device
        )


class hyper_net(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        (self.model, self.preprocess, self.tokenizer) = get_convnet(args, pretrained)
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        self.args = args
        self.freeze(self.model)
        self.Image_Adapter = nn.ModuleList()
        self.Text_Adapter = nn.ModuleList()
        self.class_mean_list = []
        self.class_cov_list = []
        embed_dim = self.feature_dim
        self.curv = torch.tensor(args["curvature"])
        self.task_id = -1
        self.label2task = {}
        self.learned_lorentz_exp = LearnableLorentzExp(dim=embed_dim)
        self.nullspace_energy_threshold = args.get("nullspace_energy_threshold", 0.95)
        self.logit_scale = self.model.logit_scale

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype),
                x,
            ],
            dim=1,
        )
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)
        if self.visual.attn_pool is not None:
            if self.visual.attn_pool_contrastive is not None:
                x = self.visual.ln_post(x)
                tokens = self.visual.attn_pool(x)
                if self.visual.attn_pool_type == "parallel":
                    pooled = self.visual.attn_pool_contrastive(x)
                else:
                    assert self.visual.attn_pool_type == "cascade"
                    pooled = self.visual.attn_pool_contrastive(tokens)
            else:
                x = self.visual.attn_pool(x)
                x = self.visual.ln_post(x)
                (pooled, tokens) = self.visual._global_pool(x)
        elif self.visual.final_ln_after_pool:
            (pooled, tokens) = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        else:
            x = self.visual.ln_post(x)
            (pooled, tokens) = self.visual._global_pool(x)
        return pooled

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [
                        weight,
                        torch.zeros(nb_classes - nb_output, self.feature_dim).cuda(),
                    ]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def update_task(self):
        if len(self.Image_Adapter) > 0:
            self.freeze(self.Image_Adapter[-1])
            self.freeze(self.Text_Adapter[-1])
        self.Image_Adapter.append(AMLP_Adapter(512, 512))
        self.Text_Adapter.append(AMLP_Adapter(512, 512))
        self.task_id += 1

    def compute_cov(
        self, known_classes, total_classes, train_loader, device, data_manager
    ):
        with torch.no_grad():
            vecs = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                (inputs, targets) = (inputs.to(device), targets.to(device))
                image_features = self.model.encode_image(inputs)
                image_features = self.Image_encode(image_features)
                vecs.append(image_features)
                labels.append(targets)
            vecs = torch.cat(vecs)
            labels = torch.cat(labels)
            cov = vecs.T @ vecs / vecs.shape[0]
            if self.learned_lorentz_exp:
                self.learned_lorentz_exp.update_covariance(cov, vecs.shape[0])
                self.learned_lorentz_exp.compute_null_space(
                    self.nullspace_energy_threshold
                )

    def Image_encode(self, image_features):
        image_res = []
        for i in range(len(self.Image_Adapter)):
            image_res.append(self.Image_Adapter[i](image_features))
        image_res = torch.sum(torch.stack(image_res), dim=0)
        return image_res

    def Text_encode(self, text_features):
        text_res = []
        for i in range(len(self.Text_Adapter)):
            text_res.append(self.Text_Adapter[i](text_features))
        text_res = torch.sum(torch.stack(text_res), dim=0)
        return text_res

    @property
    def feature_dim(self):
        return self.model.out_dim

    def encode_image(self, x, normalize=False):
        image_feats = self.model.encode_image(x, normalize=normalize)
        image_feats = self.Image_encode(image_feats)
        image_feats = self.trans_hyper(image_feats)
        return image_feats

    def encode_text(self, x, normalize=False):
        text_feats = self.model.encode_text(x, normalize=normalize)
        text_feats = self.Text_encode(text_feats)
        text_feats = self.trans_hyper(text_feats)
        return text_feats

    def trans_hyper(self, euc_x):
        return self.learned_lorentz_exp(euc_x)

    def get_curv(self) -> torch.Tensor:
        return self.curv

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
