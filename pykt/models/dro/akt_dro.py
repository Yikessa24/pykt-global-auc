import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F

from ..akt import Architecture


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AKT_DRO(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
                 kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False,
                 l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, n_skill=None,
                 dro_gamma=2.0, dro_lambda=0.7, dro_reward_alpha=0.5,
                 dro_cluster_aware=True, dro_num_clusters=4, dro_pair_temp=0.2,
                 dro_margin=0.0, dro_max_pairs=4096, dro_loss_weight=0.2,
                 dro_bad_th=0.4, dro_good_th=0.7, dro_chaotic_th=0.35,
                 dro_entropy_weight=1e-3, dro_value_weight=5e-2):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        # Use dedicated model name and register it in train/eval dispatch lists.
        self.model_name = "akt_dro"

        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = "akt"
        self.separate_qa = separate_qa
        self.emb_type = emb_type

        # DRO hyper-parameters
        self.dro_gamma = dro_gamma
        self.dro_lambda = dro_lambda
        self.dro_reward_alpha = dro_reward_alpha
        self.dro_cluster_aware = dro_cluster_aware
        self.dro_num_clusters = dro_num_clusters
        self.dro_pair_temp = dro_pair_temp
        self.dro_margin = dro_margin
        self.dro_max_pairs = dro_max_pairs
        self.dro_loss_weight = dro_loss_weight
        self.dro_bad_th = dro_bad_th
        self.dro_good_th = dro_good_th
        self.dro_chaotic_th = dro_chaotic_th
        self.dro_entropy_weight = dro_entropy_weight
        self.dro_value_weight = dro_value_weight

        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            emb_type=self.emb_type,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        # DRO state / policy / value / reward modules
        self.dro_state_proj = nn.Sequential(
            nn.Linear(d_model + embed_l, d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.dro_policy_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )
        self.dro_value_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.dro_reward_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.register_buffer(
            "dro_action_values",
            torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32),
        )
        self.register_buffer(
            "dro_pair_reward_matrix",
            torch.tensor(
                [
                    [1.0, 1.2, 1.2, 1.6],
                    [0.9, 1.0, 1.1, 1.4],
                    [0.9, 1.0, 1.0, 1.3],
                    [0.8, 0.9, 0.9, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

        self.init_params()

    def init_params(self):
        self.reset()
        for n, m in self.named_modules():
            if not n.startswith("dro_"):
                continue
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def _infer_cluster_ids(self, target, valid_mask):
        """
        Infer student-state clusters online by historical performance.
        0: bad, 1: mid_stable, 2: mid_chaotic, 3: good
        """
        target_f = target.float()
        valid_f = valid_mask.float()

        # history up to t-1
        hist_sum = torch.cumsum(target_f * valid_f, dim=1) - target_f * valid_f
        hist_cnt = torch.cumsum(valid_f, dim=1) - valid_f
        hist_acc = hist_sum / torch.clamp(hist_cnt, min=1.0)

        # response volatility proxy
        prev = torch.zeros_like(target_f)
        prev[:, 1:] = target_f[:, :-1]
        switch = (target_f - prev).abs() * valid_f
        if switch.size(1) > 1:
            win = min(5, switch.size(1))
            switch = F.avg_pool1d(
                switch.unsqueeze(1),
                kernel_size=win,
                stride=1,
                padding=win // 2,
            ).squeeze(1)
            switch = switch[:, :target_f.size(1)]

        cluster_ids = torch.full_like(target, 1, dtype=torch.long)
        bad_mask = hist_acc < self.dro_bad_th
        good_mask = hist_acc >= self.dro_good_th
        mid_mask = (~bad_mask) & (~good_mask)
        chaotic_mask = mid_mask & (switch >= self.dro_chaotic_th)
        stable_mask = mid_mask & (~chaotic_mask)

        cluster_ids[bad_mask] = 0
        cluster_ids[stable_mask] = 1
        cluster_ids[chaotic_mask] = 2
        cluster_ids[good_mask] = 3
        return cluster_ids

    def _sample_pos_neg(self, pos_idx, neg_idx):
        num_pos, num_neg = pos_idx.numel(), neg_idx.numel()
        if num_pos == 0 or num_neg == 0:
            return pos_idx, neg_idx

        total_pairs = num_pos * num_neg
        if total_pairs <= self.dro_max_pairs:
            return pos_idx, neg_idx

        keep_pos = min(num_pos, max(1, int(math.sqrt(self.dro_max_pairs))))
        keep_neg = min(num_neg, max(1, self.dro_max_pairs // keep_pos))

        pos_perm = torch.randperm(num_pos, device=pos_idx.device)[:keep_pos]
        neg_perm = torch.randperm(num_neg, device=neg_idx.device)[:keep_neg]
        return pos_idx[pos_perm], neg_idx[neg_perm]

    def _dro_ranking_loss(self, preds, target, state_repr, cluster_ids, valid_mask):
        valid = valid_mask.view(-1)
        if valid.sum() < 2:
            return preds.new_zeros(())

        flat_pred = preds.view(-1)[valid]
        flat_t = target.float().view(-1)[valid]
        flat_state = state_repr.view(-1, state_repr.size(-1))[valid]
        flat_cluster = cluster_ids.view(-1)[valid]

        pos_mask = flat_t > 0.5
        neg_mask = flat_t <= 0.5
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return preds.new_zeros(())

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]
        pos_idx, neg_idx = self._sample_pos_neg(pos_idx, neg_idx)

        pos_scores = flat_pred[pos_idx]
        neg_scores = flat_pred[neg_idx]
        pos_state = flat_state[pos_idx]
        neg_state = flat_state[neg_idx]
        pos_cluster = flat_cluster[pos_idx]
        neg_cluster = flat_cluster[neg_idx]

        pnum, nnum = pos_scores.size(0), neg_scores.size(0)
        if pnum == 0 or nnum == 0:
            return preds.new_zeros(())

        # pairwise ranking loss
        score_diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        pair_loss = F.softplus((self.dro_margin - score_diff) / max(self.dro_pair_temp, 1e-6))

        # cluster-aware base reward
        pair_pos_cluster = pos_cluster.unsqueeze(1).expand(pnum, nnum).reshape(-1)
        pair_neg_cluster = neg_cluster.unsqueeze(0).expand(pnum, nnum).reshape(-1)
        if self.dro_cluster_aware:
            base_reward = self.dro_pair_reward_matrix[pair_pos_cluster, pair_neg_cluster]
        else:
            base_reward = torch.ones_like(pair_pos_cluster, dtype=pair_loss.dtype)

        # dynamic reward optimization: state -> policy/value/reward
        pos_state_ex = pos_state.unsqueeze(1).expand(pnum, nnum, -1).reshape(-1, pos_state.size(-1))
        neg_state_ex = neg_state.unsqueeze(0).expand(pnum, nnum, -1).reshape(-1, neg_state.size(-1))
        pair_state = torch.cat([pos_state_ex, neg_state_ex, pos_state_ex - neg_state_ex], dim=-1)

        policy_logits = self.dro_policy_net(pair_state)
        policy_prob = F.softmax(policy_logits, dim=-1)
        action_scale = (policy_prob * self.dro_action_values.unsqueeze(0)).sum(dim=-1)

        value_score = self.dro_value_net(pair_state).squeeze(-1)
        reward_score = self.dro_reward_net(pair_state).squeeze(-1)
        reward_delta = torch.sigmoid(reward_score - value_score)

        dynamic_weight = base_reward * (1.0 + self.dro_reward_alpha * reward_delta) * action_scale
        dynamic_weight = torch.clamp(dynamic_weight, min=0.2, max=5.0)

        weighted_pair_loss = pair_loss.reshape(-1) * dynamic_weight

        # group DRO over cluster-pair risks
        pair_gid = pair_pos_cluster * self.dro_num_clusters + pair_neg_cluster
        group_losses = []
        for gid in range(self.dro_num_clusters * self.dro_num_clusters):
            gmask = pair_gid == gid
            if gmask.any():
                group_losses.append(weighted_pair_loss[gmask].mean())

        if len(group_losses) == 0:
            return preds.new_zeros(())

        group_losses = torch.stack(group_losses)
        if self.dro_gamma > 1e-6:
            robust_risk = torch.logsumexp(self.dro_gamma * group_losses, dim=0) / self.dro_gamma
        else:
            robust_risk = group_losses.max()
        avg_risk = group_losses.mean()
        dro_risk = self.dro_lambda * robust_risk + (1.0 - self.dro_lambda) * avg_risk

        # value learning and policy entropy regularization
        value_target = pair_loss.reshape(-1).detach()
        value_loss = F.mse_loss(value_score, value_target)
        entropy = -(policy_prob * torch.log(policy_prob + 1e-12)).sum(dim=-1).mean()

        return dro_risk + self.dro_value_weight * value_loss - self.dro_entropy_weight * entropy

    def get_loss(self, preds, target, valid_mask=None, cluster_ids=None, state_repr=None, preloss=0.0):
        if valid_mask is None:
            valid_mask = target >= 0

        y = torch.masked_select(preds, valid_mask)
        t = torch.masked_select(target, valid_mask).float()
        if y.numel() == 0:
            bce_loss = preds.new_zeros(())
        else:
            bce_loss = F.binary_cross_entropy(y, t)

        if state_repr is None:
            dro_loss = preds.new_zeros(())
        else:
            if cluster_ids is None:
                cluster_ids = self._infer_cluster_ids(target, valid_mask)
            dro_loss = self._dro_ranking_loss(preds, target, state_repr, cluster_ids, valid_mask)

        return bce_loss + preloss + self.dro_loss_weight * dro_loss

    def forward(self, q_data, target, pid_data=None, qtest=False, cluster_data=None, valid_mask=None):
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)

        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        # Add DRO regularizer only during training.
        if self.training and self.dro_loss_weight > 0:
            if valid_mask is None:
                valid_mask = target >= 0
            state_repr = self.dro_state_proj(concat_q)
            if cluster_data is None:
                cluster_ids = self._infer_cluster_ids(target, valid_mask)
            else:
                cluster_ids = cluster_data.long()
            dro_loss = self._dro_ranking_loss(preds, target, state_repr, cluster_ids, valid_mask)
        else:
            dro_loss = preds.new_zeros(())

        total_reg_loss = c_reg_loss + self.dro_loss_weight * dro_loss

        if not qtest:
            return preds, total_reg_loss
        else:
            return preds, total_reg_loss, concat_q


# Dynamic registration hint for loader utilities.
model_name = "akt_dro"
