import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


class CGCSNetLoss(nn.Module):
    """Unsupervised loss for CGCS-Net.

    Components:
      - recon: token-level L1 reconstruction (anchor)
      - co: masked commonality alignment (1-m) * ||c_x - c_y||^2
      - str: masked structural alignment on edges (1-m_edge) * |cos(sx_i,sx_j) - cos(sy_i,sy_j)|
      - mask_l1: encourage sparse changes
      - mask_tv: encourage smooth mask on graph
      - mask_ent: encourage non-collapsed mask (maximize entropy)
    """

    def __init__(self,
                 w_recon: float = 1.0,
                 w_co: float = 1.0,
                 w_str: float = 0.5,
                 w_mask_l1: float = 0.05,
                 w_mask_tv: float = 0.20,
                 w_mask_ent: float = 0.10,
                 mask_warmup: int = 10,
                 mask_ramp: int = 5):
        super().__init__()
        self.w_recon = float(w_recon)
        self.w_co = float(w_co)
        self.w_str = float(w_str)
        self.w_mask_l1 = float(w_mask_l1)
        self.w_mask_tv = float(w_mask_tv)
        self.w_mask_ent = float(w_mask_ent)
        self.mask_warmup = int(mask_warmup)
        self.mask_ramp = int(mask_ramp)

    def _gate(self, epoch: int) -> float:
        if epoch is None:
            return 1.0
        e = int(epoch)
        if e < self.mask_warmup:
            return 0.0
        if self.mask_ramp <= 0:
            return 1.0
        return float(min(1.0, (e - self.mask_warmup) / float(self.mask_ramp)))

    def forward(self,
                out: dict,
                x_in: torch.Tensor,
                y_in: torch.Tensor,
                edge_index_undir: torch.Tensor,
                edge_weight: torch.Tensor | None = None,
                epoch: int | None = None):
        # recon
        recon_x = out["recon_x"]
        recon_y = out["recon_y"]
        loss_recon = (F.l1_loss(recon_x, x_in.float()) + F.l1_loss(recon_y, y_in.float())) * 0.5

        # mask gating schedule
        g = self._gate(epoch)
        if g <= 0.0:
            m = torch.zeros_like(out["m"])
        else:
            m = out["m"].clamp(0.0, 1.0)

        # common alignment
        c_x = out["c_x"]
        c_y = out["c_y"]
        diff2 = torch.sum((c_x - c_y) ** 2, dim=1)
        loss_co = torch.mean((1.0 - m) * diff2)

        # structural alignment (edge-based, uses structure branch)
        s_x = out["s_x"]
        s_y = out["s_y"]

        if edge_index_undir is None or edge_index_undir.numel() == 0:
            loss_str = torch.zeros((), device=c_x.device)
            loss_tv = torch.zeros((), device=c_x.device)
        else:
            i = edge_index_undir[:, 0].long()
            j = edge_index_undir[:, 1].long()

            sx = F.cosine_similarity(s_x[i], s_x[j], dim=-1, eps=1e-6)
            sy = F.cosine_similarity(s_y[i], s_y[j], dim=-1, eps=1e-6)
            e = torch.abs(sx - sy)  # [E]

            m_e = 0.5 * (m[i] + m[j])
            gate_e = (1.0 - m_e)

            if edge_weight is not None and edge_weight.numel() == e.numel():
                w = edge_weight.clamp(min=0.0)
            else:
                w = torch.ones_like(e)

            loss_str = torch.mean(w * gate_e * e)

            # graph TV for mask smoothness
            loss_tv = torch.mean(w * (m[i] - m[j]) ** 2)

        # mask regularizers
        loss_l1 = torch.mean(m)
        # maximize entropy -> minimize negative entropy
        ent = _safe_entropy(m)
        loss_ent = -torch.mean(ent)

        # total
        loss = (self.w_recon * loss_recon
                + g * self.w_co * loss_co
                + g * self.w_str * loss_str
                + self.w_mask_l1 * loss_l1
                + self.w_mask_tv * loss_tv
                + self.w_mask_ent * loss_ent)

        # DI proxy for logging: cosine distance in common space
        di = 1.0 - F.cosine_similarity(c_x, c_y, dim=-1, eps=1e-6)
        di_mean = float(di.detach().mean().item())

        stats = {
            "loss_recon": float(loss_recon.detach().item()),
            "loss_co": float((g * loss_co).detach().item()),
            "loss_str": float((g * loss_str).detach().item()),
            "loss_mask_l1": float((self.w_mask_l1 * loss_l1).detach().item()),
            "loss_mask_tv": float((self.w_mask_tv * loss_tv).detach().item()),
            "loss_mask_ent": float((self.w_mask_ent * loss_ent).detach().item()),
            "m_mean": float(m.detach().mean().item()),
            "di_mean": di_mean,
        }
        # keep compatible return signature
        return loss, stats, di.detach(), None
