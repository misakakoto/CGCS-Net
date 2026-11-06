import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cosine distance in [0, 2], larger means more different. Shape: [K]."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)


class GatedMP(nn.Module):
    """Low-memory gated message passing on a directed edge list (i <- j)."""

    def __init__(self, dim: int, drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.lin_z = nn.Linear(dim * 2, dim)
        self.lin_h = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor,
                edge_index_dir: torch.Tensor | None,
                edge_weight_dir: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [K,D]
        edge_index_dir: [E,2] with columns [dst(i), src(j)]
        edge_weight_dir: [E] optional
        """
        if edge_index_dir is None or edge_index_dir.numel() == 0:
            return x

        dst = edge_index_dir[:, 0].long()
        src = edge_index_dir[:, 1].long()

        msg = x[src]
        if edge_weight_dir is not None and edge_weight_dir.numel() == msg.shape[0]:
            msg = msg * edge_weight_dir.view(-1, 1).clamp(min=0.0)

        K, D = x.shape
        agg = torch.zeros((K, D), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((K,), device=x.device, dtype=x.dtype)
        if edge_weight_dir is None or edge_weight_dir.numel() != msg.shape[0]:
            deg.index_add_(0, dst, torch.ones((msg.shape[0],), device=x.device, dtype=x.dtype))
        else:
            deg.index_add_(0, dst, edge_weight_dir.clamp(min=0.0))
        agg = agg / (deg.view(-1, 1) + 1e-6)

        inp = torch.cat([x, agg], dim=1)
        z = torch.sigmoid(self.lin_z(inp))
        h = torch.tanh(self.lin_h(inp))
        out = (1.0 - z) * x + z * h
        out = self.norm(out)
        out = self.drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(drop)),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CGCSNetModel(nn.Module):
    """CGCS-Net: Change-Guided Commonality-Structure model (token-level, graph-based).

    Inputs:
      x_tok: [K,Fx], y_tok: [K,Fy], pos: [K,2]
      edge_index_dir: directed edges (i<-j), shape [E,2]
      edge_weight_dir: optional directed edge weights [E]
    Outputs dict:
      c_x, c_y: common embeddings [K,Dc]
      s_x, s_y: structure embeddings [K,Ds]
      recon_x, recon_y: token reconstructions
      m: mask probability [K] (suspected change)
    """

    def __init__(self, in_x: int, in_y: int, hidden_dim: int = 128, depth: int = 4, drop: float = 0.1,
                 c_dim: int | None = None, s_dim: int | None = None):
        super().__init__()
        self.in_x = int(in_x)
        self.in_y = int(in_y)
        self.hidden = int(hidden_dim)
        self.depth = int(depth)
        self.drop = float(drop)

        c_dim = int(c_dim) if c_dim is not None else self.hidden
        s_dim = int(s_dim) if s_dim is not None else self.hidden

        self.enc_x = nn.Sequential(nn.Linear(self.in_x, self.hidden), nn.GELU(), nn.LayerNorm(self.hidden))
        self.enc_y = nn.Sequential(nn.Linear(self.in_y, self.hidden), nn.GELU(), nn.LayerNorm(self.hidden))

        self.mp_layers = nn.ModuleList([GatedMP(self.hidden, drop=self.drop) for _ in range(self.depth)])

        # shared projections (force a common/structure factorization)
        self.proj_c = nn.Sequential(nn.Linear(self.hidden, c_dim), nn.GELU(), nn.LayerNorm(c_dim))
        self.proj_s = nn.Sequential(nn.Linear(self.hidden, s_dim), nn.GELU(), nn.LayerNorm(s_dim))

        # decoders (token reconstruction, acts as an anchor to prevent collapse)
        self.dec_x = MLP(self.hidden, max(64, self.hidden), self.in_x, drop=self.drop)
        self.dec_y = MLP(self.hidden, max(64, self.hidden), self.in_y, drop=self.drop)

        # mask head input:
        # [|c_x-c_y| (Dc), |s_x-s_y| (Ds), recon_err (1), pos (2)]
        mask_in = c_dim + s_dim + 1 + 2
        self.mask_head = nn.Sequential(
            nn.Linear(mask_in, max(64, self.hidden // 2)),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(max(64, self.hidden // 2), 1),
        )

    def forward(self,
                x_tok: torch.Tensor,
                y_tok: torch.Tensor,
                pos: torch.Tensor,
                edge_index_dir: torch.Tensor | None = None,
                edge_weight_dir: torch.Tensor | None = None,
                edge_index_undir: torch.Tensor | None = None,
                edge_weight: torch.Tensor | None = None) -> dict:

        hx = self.enc_x(x_tok.float())
        hy = self.enc_y(y_tok.float())

        for layer in self.mp_layers:
            hx = layer(hx, edge_index_dir=edge_index_dir, edge_weight_dir=edge_weight_dir)
            hy = layer(hy, edge_index_dir=edge_index_dir, edge_weight_dir=edge_weight_dir)

        # recon (anchor)
        recon_x = self.dec_x(hx)
        recon_y = self.dec_y(hy)

        c_x = self.proj_c(hx)
        c_y = self.proj_c(hy)
        s_x = self.proj_s(hx)
        s_y = self.proj_s(hy)

        # recon error (scalar per node)
        rx = torch.mean(torch.abs(recon_x - x_tok.float()), dim=1, keepdim=True)
        ry = torch.mean(torch.abs(recon_y - y_tok.float()), dim=1, keepdim=True)
        rerr = 0.5 * (rx + ry)

        # per-dimension diffs for mask prediction
        dc = torch.abs(c_x - c_y)  # [K, Dc]
        ds = torch.abs(s_x - s_y)  # [K, Ds]

        pos = pos.float()
        if pos.dim() == 1:
            pos = pos.view(-1, 1)

        m_in = torch.cat([dc, ds, rerr, pos], dim=1)
        m = torch.sigmoid(self.mask_head(m_in)).squeeze(1)

        return {
            "c_x": c_x, "c_y": c_y,
            "s_x": s_x, "s_y": s_y,
            "recon_x": recon_x, "recon_y": recon_y,
            "m": m,
        }
