import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- 1. 线性注意力模块 ---
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        q = q * self.scale
        attn_out = (q @ k.transpose(-2, -1)) @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(attn_out)


# --- 2. 多 patch 实例分类器 ---
class InstanceClassifier_MultiPatchAttn(nn.Module):
    def __init__(self, in_dim, out_dim=2, dropout=0.25):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.attn = LinearAttention(dim=in_dim, heads=8, dim_head=in_dim // 8, dropout=dropout)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [N_total, D]
        residual = x
        x = x.unsqueeze(0)              # [1, N_total, D]
        x = self.norm(x)
        x = self.attn(x)
        x = x.squeeze(0)
        x = x + residual
        return self.fc(x)


# --- 3. 每类 top-K refine 模块 ---
class RefineAttnBlock(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.attn = LinearAttention(dim=in_dim, heads=8, dim_head=in_dim // 8, dropout=dropout)

    def forward(self, x):
        # x: [K, D]
        residual = x
        x = x.unsqueeze(0)
        x = self.norm(x)
        x = self.attn(x)
        x = x.squeeze(0)
        x = x + residual
        return x.mean(dim=0)


# --- 4. 主模型 ---
class EffusionMIL(nn.Module):
    def __init__(self, gate=True, size_arg="big", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
                 act=nn.ReLU(), instance_eval=False,
                 refine_topk=None):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]

        if isinstance(act, str):
            act_lower = act.lower()
            if act_lower == 'relu':
                act_layer = nn.ReLU()
            elif act_lower == 'silu':
                act_layer = nn.SiLU()
            elif act_lower in ['gelu', 'gule']:
                act_layer = nn.GELU()
            else:
                act_layer = nn.ReLU()
        else:
            act_layer = act

        self.trans = nn.Sequential(
            nn.Linear(size[0], size[1]),
            act_layer,
            nn.Dropout(dropout)
        )

        if gate:
            self.attention_net = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
            self.attention_gate = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            )
            self.attention_c = nn.Linear(size[2], n_classes)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(size[2], n_classes)
            )

        self.gate = gate
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])

        # 多 patch instance 分类器
        self.instance_classifiers = nn.ModuleList([
            InstanceClassifier_MultiPatchAttn(in_dim=size[1], out_dim=2, dropout=dropout)
            for _ in range(n_classes)
        ])

        # 推理 refine 模块
        self.refine_blocks = nn.ModuleList([
            RefineAttnBlock(in_dim=size[1], dropout=dropout)
            for _ in range(n_classes)
        ])
        self.refine_alpha = nn.Parameter(torch.zeros(n_classes))

        self.k_sample = k_sample
        self.refine_topk = refine_topk if refine_topk is not None else 2 * k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def compute_attention(self, h):
        if self.gate:
            a = self.attention_net(h)
            b = self.attention_gate(h)
            A = a.mul(b)
            A = self.attention_c(A)
        else:
            A = self.attention_net(h)
        return A

    def inst_eval(self, A, h, classifier):
        """实例分类路径。"""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        sample_num = min(self.k_sample, A.shape[1])

        top_p_ids = torch.topk(A, sample_num)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, sample_num, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self.create_positive_targets(sample_num, device)
        n_targets = self.create_negative_targets(sample_num, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)  # [2k, D]

        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        sample_num = min(self.k_sample, A.shape[1])

        top_p_ids = torch.topk(A, sample_num)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(sample_num, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def _bag_refine(self, A_raw, h):
        N = h.shape[0]
        K = min(self.refine_topk, N)
        refined = []
        for c in range(self.n_classes):
            top_ids = torch.topk(A_raw[c], K)[1]
            h_top = torch.index_select(h, 0, top_ids)
            refined.append(self.refine_blocks[c](h_top))
        return torch.stack(refined, dim=0)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        if len(h.shape) == 3:
            h = h.squeeze(0)

        h = self.trans(h)

        A = self.compute_attention(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        results_dict = {}
        total_inst_loss = 0.0
        do_instance_eval = instance_eval or self.instance_eval

        if do_instance_eval:
            all_preds, all_targets = [], []
            if label is not None:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label).to(device)
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
                for i in range(self.n_classes):
                    classifier = self.instance_classifiers[i]
                    current_label = inst_labels[i] if inst_labels.dim() > 0 else inst_labels

                    if current_label.item() == 1:
                        instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            instance_loss = 0.0
                            continue
                    total_inst_loss += instance_loss
                if self.subtyping:
                    total_inst_loss /= self.n_classes

            results_dict['instance_loss'] = total_inst_loss
            results_dict['inst_labels'] = np.array(all_targets) if len(all_targets) > 0 else np.array([])
            results_dict['inst_preds'] = np.array(all_preds) if len(all_preds) > 0 else np.array([])
        else:
            results_dict['instance_loss'] = 0.0

        # ---- 训推一致融合 ----
        M = torch.mm(A, h)
        M_refine = self._bag_refine(A_raw, h)
        alpha = self.refine_alpha.unsqueeze(-1)
        M_final = M + alpha * M_refine

        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M_final[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if return_features:
            results_dict['features'] = M_final
        results_dict['logits'] = logits
        results_dict['Y_prob'] = Y_prob
        results_dict['Y_hat'] = Y_hat
        return results_dict
