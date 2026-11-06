import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
try:
    from torch import amp as torch_amp
except Exception:
    torch_amp = None
import torch.nn.functional as F
from scipy.io import loadmat
from skimage import io
from skimage.segmentation import slic, mark_boundaries

import cv2

from evaluation import Evaluation
from model_cgcs_net import CGCSNetModel, cosine_distance as cosine_distance_cgcs
from loss_cgcs_net import CGCSNetLoss
from preprocess import process_images


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CGCS-Net: Change-Guided Commonality-Structure Network")

    # I/O（输入输出）
    p.add_argument('--root_dir', type=str, default=r'.')  # 项目根目录，会与 load_data_dir 拼接成数据路径
    p.add_argument('--seed', type=int, default=2988502036)  # 固定随机种子值（seed_mode=fixed 时生效）
    p.add_argument('--seed_mode', type=str, default='fixed', choices=['fixed', 'random'])
    p.add_argument('--oracle_best_f1', action='store_true', help='Evaluation-only: select best checkpoint using ground-truth F1 (not label-free).')
    p.add_argument('--load_data_dir', type=str, default='data/LPEM')  # 数据集目录（相对 root_dir）
    p.add_argument('--mat_file', type=str, default='dataset#14.mat')  # .mat 数据文件名
    p.add_argument('--result_folder', type=str, default=None)  # 结果输出目录；None 时自动设为 results/<mat文件名>

    # Data type（影像类型）
    p.add_argument('--img_t1_type1', type=str, default='opt', choices=['opt', 'sar'])  # T1 影像类型：光学(opt)或雷达(sar)
    p.add_argument('--img_t2_type2', type=str, default='opt', choices=['opt', 'sar'])  # T2 影像类型：光学(opt)或雷达(sar)

    # Superpixels（超像素）
    p.add_argument('--N_SEG', type=int, default=2100)  # SLIC 超像素数量
    p.add_argument('--Com', type=float, default=36)  # SLIC 紧凑度（compactness）
    p.add_argument('--sp_on', type=str, default='avg', choices=['t1', 't2', 'avg'])  # 用哪一幅图做超像素：T1/T2/两者平均
    p.add_argument('--sp_feat', type=str, default='mean', choices=['mean', 'meanstd'])  # 超像素特征：均值 或 均值+标准差
    p.add_argument('--slic_sigma', type=float, default=1.0)  # SLIC 前高斯平滑强度
    p.add_argument('--slic_zero', action='store_true')  # 是否启用 SLICO（零参数变体）

    # CGCS-Net model（模型与损失权重）
    p.add_argument('--cgcs_dim', type=int, default=128, help='CGCS-Net hidden dimension')  # 隐层维度
    p.add_argument('--cgcs_depth', type=int, default=4, help='CGCS-Net graph message-passing depth')  # 图消息传递层数
    p.add_argument('--cgcs_drop', type=float, default=0.10, help='CGCS-Net dropout')  # Dropout 比例
    p.add_argument('--cgcs_w_recon', type=float, default=1.00, help='CGCS-Net reconstruction loss weight')  # 重建损失权重
    p.add_argument('--cgcs_w_co', type=float, default=0.20, help='CGCS-Net masked commonality alignment weight')  # 公共性对齐损失权重
    p.add_argument('--cgcs_w_str', type=float, default=0.80, help='CGCS-Net masked structural alignment weight')  # 结构对齐损失权重
    p.add_argument('--cgcs_w_mask_l1', type=float, default=0.20, help='CGCS-Net mask sparsity (L1) weight')  # 掩码稀疏约束（L1）权重
    p.add_argument('--cgcs_w_mask_tv', type=float, default=0.20, help='CGCS-Net mask smoothness (graph TV) weight')  # 掩码平滑约束（图 TV）权重
    p.add_argument('--cgcs_w_mask_ent', type=float, default=0.10, help='CGCS-Net mask entropy (anti-collapse) weight')  # 掩码熵正则权重（防塌缩）
    p.add_argument('--cgcs_mask_warmup', type=int, default=10, help='Warmup epochs before enabling mask gating')  # 启用 mask gating 前的预热轮数
    p.add_argument('--cgcs_eta_str', type=float, default=0.30, help='DI fusion: DI_com + eta*DI_str')  # DI 融合中结构项系数 eta
    p.add_argument('--cgcs_kappa_mask', type=float, default=0.00, help='DI boost by mask: *(1 + kappa*DI_mask)')  # 按 DI_mask 放大 DI 的系数 kappa
    p.add_argument('--cgcs_di_mask_mix', type=float, default=0.00, help='DI fusion: (1-mix)*DI + mix*DI_mask')  # DI 与 DI_mask 的线性混合比例
    p.add_argument('--cgcs_ctx_mix', type=float, default=0.60, help='DI_str context mix ratio for CGCS-Net')  # DI_str 中边差异与上下文差异的混合比例

    # Graph（图结构）
    p.add_argument('--edge_knn_k', type=int, default=0, help='Add kNN edges on superpixel centroids (0=disable)')  # 额外添加的 kNN 边数（0 表示不加）

    # Training（训练）
    p.add_argument('--epochs', type=int, default=70)  # 训练轮数
    p.add_argument('--learning_rate', type=float, default=1e-4)  # 学习率

    # Edge weights and DI smoothing（边权与 DI 平滑）
    p.add_argument('--edge_weight', type=str, default='gaussian', choices=['none', 'gaussian'])  # 边权模式：不加权/高斯相似度加权
    p.add_argument('--edge_sigma', type=float, default=-1.0)  # 高斯边权的 sigma；<=0 时使用中位数启发式
    p.add_argument('--di_str_node_min', type=float, default=0.35, help='Min clamp for node_weight in DI_str.')  # DI_str 节点权重下限截断
    p.add_argument('--di_str_node_max', type=float, default=3.0, help='Max clamp for node_weight in DI_str.')  # DI_str 节点权重上限截断
    p.add_argument('--di_smooth_alpha', type=float, default=0.5)  # 图平滑强度 alpha（越大平滑越强）
    p.add_argument('--di_smooth_iter', type=int, default=0)  # 图平滑迭代次数（0 表示不平滑）

    # DI normalization（DI 归一化）
    p.add_argument('--clip_p1', type=float, default=1.0)  # DI 百分位裁剪下界（P1）
    p.add_argument('--clip_p99', type=float, default=99.0)  # DI 百分位裁剪上界（P99）
    p.add_argument('--save_di_parts', type=int, default=1, choices=[0, 1])  # 是否保存 DI_com/DI_str/DI_mask 可视化

    # Post-process (CM)（后处理）
    p.add_argument('--cm_th_mode', type=str, default='otsu', choices=['otsu', 'wotsu'])  # 阈值模式：普通 Otsu / 加权 Otsu
    p.add_argument('--tau_eval', type=float, default=0.30, help='Used when cm_th_mode=wotsu')  # wOtsu 中指数权重衰减系数
    p.add_argument('--blur_sigma', type=float, default=0.0)  # 阈值分割前的高斯模糊强度
    p.add_argument('--open_ks', type=int, default=3)  # 开运算核大小（去小噪点）
    p.add_argument('--close_ks', type=int, default=3)  # 闭运算核大小（填小孔洞）
    p.add_argument('--open_it', type=int, default=1)  # 开运算迭代次数
    p.add_argument('--close_it', type=int, default=1)  # 闭运算迭代次数
    p.add_argument('--min_area', type=int, default=600)  # 连通域最小面积，小于该值会被删除

    return p


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_run_hparams(args: argparse.Namespace, out_dir: str) -> tuple[str, str]:
    """Save run hyperparameters for reproducibility.
    Returns (latest_json_path, timestamped_json_path).
    """
    ensure_dir(out_dir)
    now = datetime.now()
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    payload = {
        "run_id": run_id,
        "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "args": vars(args),
    }

    latest_json = os.path.join(out_dir, 'run_hparams_latest.json')
    run_json = os.path.join(out_dir, f'run_hparams_{run_id}.json')
    latest_txt = os.path.join(out_dir, 'run_hparams_latest.txt')
    run_txt = os.path.join(out_dir, f'run_hparams_{run_id}.txt')

    for fp in (latest_json, run_json):
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    lines = [
        f"run_id: {payload['run_id']}",
        f"timestamp: {payload['timestamp']}",
        f"cwd: {payload['cwd']}",
        f"command: {payload['command']}",
        "",
        "[args]",
    ]
    for k in sorted(payload["args"].keys()):
        lines.append(f"{k}: {payload['args'][k]}")
    text = "\n".join(lines) + "\n"
    for fp in (latest_txt, run_txt):
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(text)

    return latest_json, run_json


def load_images_mat(data_path: str, mat_file: str, device: torch.device):
    mat = loadmat(os.path.join(data_path, mat_file))
    a = mat['image_t1']
    b = mat['image_t2']
    ref = mat.get('Ref_gt')
    if ref is None:
        return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), None
    if ref.dtype != np.uint8:
        ref = ref.astype(np.uint8)
    if ref.max() > 1:
        ref = (ref > 0).astype(np.uint8)
    a_t = torch.from_numpy(a).to(device)
    b_t = torch.from_numpy(b).to(device)
    return a_t, b_t, ref


def _prepare_slic_image(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    if x_max - x_min > 0:
        if x_min < 0.0 or x_max <= 1.5:
            x = (x - x_min) / (x_max - x_min + 1e-6)
            x = x * 255.0
    return x


def compute_superpixels(img_ref: np.ndarray, n_segments: int, compactness: float,
                        img_type: str, sigma: float = 1.0, slic_zero: bool = False) -> np.ndarray:
    x = _prepare_slic_image(img_ref)
    convert2lab = (img_type == 'opt' and x.ndim == 3 and x.shape[-1] == 3)
    if x.ndim == 2:
        labels = slic(x, n_segments=n_segments, compactness=compactness,
                      start_label=0, channel_axis=None, convert2lab=False,
                      sigma=sigma, slic_zero=slic_zero)
    else:
        labels = slic(x, n_segments=n_segments, compactness=compactness,
                      start_label=0, channel_axis=-1, convert2lab=convert2lab,
                      sigma=sigma, slic_zero=slic_zero)
    return labels.astype(np.int32)


def build_edge_index(labels: np.ndarray) -> np.ndarray:
    H, W = labels.shape
    a = labels[:, :-1]
    b = labels[:, 1:]
    c = labels[:-1, :]
    d = labels[1:, :]

    e1 = np.stack([a[a != b], b[a != b]], axis=1) if np.any(a != b) else np.zeros((0, 2), np.int32)
    e2 = np.stack([c[c != d], d[c != d]], axis=1) if np.any(c != d) else np.zeros((0, 2), np.int32)
    edges = np.concatenate([e1, e2], axis=0)
    if edges.size == 0:
        return edges

    i = np.minimum(edges[:, 0], edges[:, 1])
    j = np.maximum(edges[:, 0], edges[:, 1])
    edges = np.stack([i, j], axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def edges_to_directed(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return edge_index
    rev = edge_index[:, ::-1]
    return np.concatenate([edge_index, rev], axis=0)


def compute_pos_counts_np(labels: np.ndarray):
    # Centroid positions and counts from superpixel labels (numpy).
    H, W = labels.shape
    lab = labels.reshape(-1).astype(np.int64)
    K = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=K).astype(np.float32)
    cnt[cnt <= 0] = 1.0
    xs = (np.tile(np.arange(W, dtype=np.float32), H) / max(1.0, float(W - 1)))
    ys = (np.repeat(np.arange(H, dtype=np.float32), W) / max(1.0, float(H - 1)))
    sumx = np.bincount(lab, weights=xs, minlength=K).astype(np.float32)
    sumy = np.bincount(lab, weights=ys, minlength=K).astype(np.float32)
    pos = np.stack([sumx / cnt, sumy / cnt], axis=1).astype(np.float32)
    return pos, cnt.reshape(-1, 1), K

def build_knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    # kNN edges on centroid positions. Returns undirected unique edges [E,2] with i<j.
    k = int(k)
    if k <= 0:
        return np.zeros((0, 2), np.int32)
    K = int(pos.shape[0])
    if K <= 1:
        return np.zeros((0, 2), np.int32)
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = (diff ** 2).sum(axis=2).astype(np.float32)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, K - 1)
    idx = np.argpartition(d2, kth=kk, axis=1)[:, :kk]
    src = np.repeat(np.arange(K, dtype=np.int32), kk)
    dst = idx.reshape(-1).astype(np.int32)
    i = np.minimum(src, dst)
    j = np.maximum(src, dst)
    e = np.stack([i, j], axis=1)
    e = np.unique(e, axis=0)
    return e.astype(np.int32)


@dataclass
class SuperpixelTokens:
    x: torch.Tensor        # [K, F]
    y: torch.Tensor        # [K, F]
    pos: torch.Tensor      # [K, 2]
    counts: torch.Tensor   # [K, 1]


def pool_tokens(img_t1: torch.Tensor, img_t2: torch.Tensor, labels: np.ndarray,
                feat_mode: str, device: torch.device) -> SuperpixelTokens:
    H, W = labels.shape
    lab = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    lab_f = lab.reshape(-1)
    K = int(lab_f.max().item()) + 1

    if img_t1.dim() == 2:
        img_t1 = img_t1.unsqueeze(-1)
        img_t2 = img_t2.unsqueeze(-1)

    feat1 = img_t1.float().reshape(-1, img_t1.shape[-1])
    feat2 = img_t2.float().reshape(-1, img_t2.shape[-1])

    counts = torch.bincount(lab_f, minlength=K).float().unsqueeze(1).clamp(min=1.0)

    sum1 = torch.zeros((K, feat1.shape[1]), device=device)
    sum2 = torch.zeros((K, feat2.shape[1]), device=device)
    sum1.index_add_(0, lab_f, feat1)
    sum2.index_add_(0, lab_f, feat2)
    mean1 = sum1 / counts
    mean2 = sum2 / counts

    if feat_mode == 'meanstd':
        sumsq1 = torch.zeros((K, feat1.shape[1]), device=device)
        sumsq2 = torch.zeros((K, feat2.shape[1]), device=device)
        sumsq1.index_add_(0, lab_f, feat1 * feat1)
        sumsq2.index_add_(0, lab_f, feat2 * feat2)
        var1 = (sumsq1 / counts) - (mean1 * mean1)
        var2 = (sumsq2 / counts) - (mean2 * mean2)
        std1 = torch.sqrt(torch.clamp(var1, min=0.0) + 1e-6)
        std2 = torch.sqrt(torch.clamp(var2, min=0.0) + 1e-6)
        tok1 = torch.cat([mean1, std1], dim=1)
        tok2 = torch.cat([mean2, std2], dim=1)
    else:
        tok1 = mean1
        tok2 = mean2

    # centroid positions in [0,1]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    xs = xs.reshape(-1) / max(1.0, float(W - 1))
    ys = ys.reshape(-1) / max(1.0, float(H - 1))
    sumx = torch.zeros((K,), device=device)
    sumy = torch.zeros((K,), device=device)
    sumx.index_add_(0, lab_f, xs)
    sumy.index_add_(0, lab_f, ys)
    cx = (sumx / counts.squeeze(1)).clamp(0.0, 1.0)
    cy = (sumy / counts.squeeze(1)).clamp(0.0, 1.0)
    pos = torch.stack([cx, cy], dim=1)

    return SuperpixelTokens(x=tok1, y=tok2, pos=pos, counts=counts)


def compute_edge_weight(tokens: torch.Tensor, edge_index: torch.Tensor, mode: str = 'gaussian', sigma: float = -1.0):
    if mode == 'none' or edge_index.numel() == 0:
        return None
    i = edge_index[:, 0].long()
    j = edge_index[:, 1].long()
    a = tokens[i]
    b = tokens[j]
    dist2 = ((a - b) ** 2).sum(dim=1)
    if sigma is None or float(sigma) <= 0:
        med = torch.median(dist2.detach())
        sigma2 = (med + 1e-6)
    else:
        sigma2 = float(sigma) ** 2
    w = torch.exp(-dist2 / (2.0 * sigma2))
    return w


def graph_smooth_scores(d: torch.Tensor, edge_index_dir: torch.Tensor, edge_weight: torch.Tensor | None,
                        alpha: float, iters: int) -> torch.Tensor:
    if edge_index_dir.numel() == 0 or iters <= 0 or alpha <= 0:
        return d
    i = edge_index_dir[:, 0].long()
    j = edge_index_dir[:, 1].long()
    if edge_weight is None:
        w = torch.ones((edge_index_dir.shape[0],), device=d.device)
    else:
        if edge_weight.numel() * 2 == edge_index_dir.shape[0]:
            w = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            w = edge_weight

    K = d.shape[0]
    wsum = torch.zeros((K,), device=d.device)
    wsum.index_add_(0, i, w)
    s = d.clone()
    for _ in range(iters):
        nbr_sum = torch.zeros((K,), device=d.device)
        nbr_sum.index_add_(0, i, w * s[j])
        s = (d + alpha * nbr_sum) / (1.0 + alpha * wsum + 1e-6)
    return s


def graph_aggregate_feat(z: torch.Tensor,
                         edge_index_dir: torch.Tensor,
                         edge_weight: torch.Tensor | None,
                         alpha: float = 1.0,
                         iters: int = 1) -> torch.Tensor:
    """Neighbor aggregation in feature space.
    z: [K,D], edge_index_dir: [2E,2] directed edges (i<-j), edge_weight: [E] (undirected)
    Returns aggregated features with the same shape [K,D].
    """
    if edge_index_dir is None or edge_index_dir.numel() == 0 or iters <= 0 or alpha <= 0:
        return z
    i = edge_index_dir[:, 0].long()
    j = edge_index_dir[:, 1].long()

    if edge_weight is None:
        w = torch.ones((edge_index_dir.shape[0],), device=z.device)
    else:
        # edge_weight is for undirected edges; directed has 2x
        if edge_weight.numel() * 2 == edge_index_dir.shape[0]:
            w = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            w = edge_weight

    K, D = z.shape
    wsum = torch.zeros((K,), device=z.device)
    wsum.index_add_(0, i, w)

    out = z.clone()
    for _ in range(int(iters)):
        nbr_sum = torch.zeros((K, D), device=z.device)
        nbr_sum.index_add_(0, i, w.view(-1, 1) * out[j])
        out = (z + alpha * nbr_sum) / (1.0 + alpha * wsum.view(-1, 1) + 1e-6)
    return out


def structure_diff_per_node(c_x: torch.Tensor,
                            c_y: torch.Tensor,
                            edge_undir: torch.Tensor,
                            edge_dir: torch.Tensor | None = None,
                            edge_weight: torch.Tensor | None = None,
                            node_weight: torch.Tensor | None = None,
                            ctx_mix: float = 0.6) -> torch.Tensor:
    """DI_str (per-node) with two complementary terms:

    1) Edge-diff term: for each undirected edge (i,j),
         e_ij = | cos(c_x[i],c_x[j]) - cos(c_y[i],c_y[j]) |
       Then distribute weighted e_ij to incident nodes (weighted average).

    2) Context-diff term: neighbor-aggregated embeddings g_x, g_y computed on directed graph,
         g_x = Agg(c_x), g_y = Agg(c_y), then
         di_ctx(i) = 1 - cos(g_x[i], g_y[i])

    Final:
       DI_str = (1-ctx_mix) * DI_edge + ctx_mix * DI_ctx

    This version is more robust than the plain mean(e_ij) because:
      - it can downweight weak/suspect edges via edge_weight
      - it can downweight low-confidence nodes via node_weight
      - context term reduces salt-and-pepper artifacts.
    """
    K = c_x.shape[0]
    if edge_undir is None or edge_undir.numel() == 0:
        return torch.zeros((K,), device=c_x.device)

    i = edge_undir[:, 0].long()
    j = edge_undir[:, 1].long()

    sx = F.cosine_similarity(c_x[i], c_x[j], dim=-1, eps=1e-6)
    sy = F.cosine_similarity(c_y[i], c_y[j], dim=-1, eps=1e-6)
    e = torch.abs(sx - sy)  # [E]

    # edge weights: similarity-based boundary preservation (optional)
    w_e = torch.ones_like(e)
    if edge_weight is not None and edge_weight.numel() == e.numel():
        w_e = w_e * edge_weight.clamp(min=0.0)

    # node weights: prefer confident/likely-unchanged nodes (optional)
    if node_weight is not None and node_weight.numel() == K:
        w_e = w_e * (0.5 * (node_weight[i] + node_weight[j])).clamp(min=0.0)

    out = torch.zeros((K,), device=c_x.device)
    deg = torch.zeros((K,), device=c_x.device)
    out.index_add_(0, i, w_e * e)
    out.index_add_(0, j, w_e * e)
    deg.index_add_(0, i, w_e)
    deg.index_add_(0, j, w_e)
    di_edge = out / (deg + 1e-6)

    # context term
    ctx_mix = float(ctx_mix)
    if ctx_mix <= 1e-6 or edge_dir is None or edge_dir.numel() == 0:
        return di_edge
    if ctx_mix >= 1.0 - 1e-6:
        ctx_mix = 1.0

    g_x = graph_aggregate_feat(c_x, edge_dir, edge_weight=edge_weight, alpha=1.0, iters=1)
    g_y = graph_aggregate_feat(c_y, edge_dir, edge_weight=edge_weight, alpha=1.0, iters=1)
    di_ctx = 1.0 - F.cosine_similarity(g_x, g_y, dim=-1, eps=1e-6)

    return (1.0 - ctx_mix) * di_edge + ctx_mix * di_ctx


def _weighted_otsu_u8(values_u8: np.ndarray, weights: np.ndarray) -> int:
    v = values_u8.reshape(-1).astype(np.int32)
    w = weights.reshape(-1).astype(np.float64)
    w = np.clip(w, 0.0, None)
    if v.size == 0:
        return 0
    hist = np.bincount(v, weights=w, minlength=256).astype(np.float64)
    if hist.sum() <= 0:
        from skimage.filters.thresholding import threshold_otsu
        return int(threshold_otsu(values_u8))
    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega) + 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    t = int(np.argmax(sigma_b2))
    return t


def postprocess_cm(di_u8: np.ndarray,
                   cm_th_mode: str,
                   weights_u8: np.ndarray | None,
                   tau_eval: float,
                   blur_sigma: float,
                   open_ks: int, close_ks: int, open_it: int, close_it: int, min_area: int) -> np.ndarray:
    from skimage.filters.thresholding import threshold_otsu

    x = di_u8
    if blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(blur_sigma))

    if cm_th_mode == 'wotsu':
        # weights: prefer low DI (likely unchanged) + area
        if weights_u8 is None:
            th = int(threshold_otsu(x))
        else:
            # compute weight = exp(-(di/255)/tau_eval) * weights_u8
            di01 = (x.astype(np.float32) / 255.0)
            w_area = weights_u8
            # make sure weights match DI map shape
            if isinstance(w_area, np.ndarray):
                if w_area.ndim == 1 and w_area.size == x.size:
                    w_area = w_area.reshape(x.shape)
                elif w_area.shape != x.shape and w_area.size == x.size:
                    w_area = w_area.reshape(x.shape)
            w = np.exp(-di01 / (tau_eval + 1e-6)).astype(np.float64) * w_area.astype(np.float64)
            th = _weighted_otsu_u8(x, w)
    else:
        th = int(threshold_otsu(x))

    cm = (x >= th).astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    if open_it > 0:
        cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, k_open, iterations=int(open_it))
    if close_it > 0:
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, k_close, iterations=int(close_it))

    if min_area > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
        out = np.zeros_like(cm)
        for idx in range(1, num):
            if stats[idx, cv2.CC_STAT_AREA] >= int(min_area):
                out[lab == idx] = 1
        cm = out

    return (cm * 255).astype(np.uint8)


def compute_metrics(label_u8: np.ndarray, pred_u8: np.ndarray) -> dict:
    if label_u8.max() <= 1 and pred_u8.max() > 1:
        label_u8 = (label_u8.astype(np.uint8) * 255)
    Indicators = Evaluation(label_u8, pred_u8)
    OA, kappa, AA = Indicators.Classification_indicators()
    Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    TP, TN, FP, FN = Indicators.matrix()
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "OA": float(OA), "Kappa": float(kappa), "AA": float(AA),
        "Precision": float(Precision), "Recall": float(Recall), "F1": float(F1),
    }


def main():
    args = build_argparser().parse_args()

    if args.seed_mode == 'random':
        run_seed = int.from_bytes(os.urandom(4), byteorder='big', signed=False)
    else:
        run_seed = int(args.seed)
    args.seed = int(run_seed)

    # reproducibility
    random.seed(int(run_seed))
    np.random.seed(int(run_seed) % (2 ** 32))
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] mode={args.seed_mode} seed={run_seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # results folder
    if args.result_folder is None:
        name = os.path.splitext(os.path.basename(args.mat_file))[0]
        args.result_folder = os.path.join('results', name)
    ensure_dir(args.result_folder)
    out_dir = os.path.join(args.result_folder, "cgcs_net_results")
    ensure_dir(out_dir)
    latest_hp_path, run_hp_path = save_run_hparams(args, out_dir)
    print(f"[RUN] hyperparameters saved: {latest_hp_path}")
    print(f"[RUN] hyperparameters archived: {run_hp_path}")


    # load
    data_path = os.path.join(args.root_dir, args.load_data_dir)
    img_t1_raw, img_t2_raw, ref = load_images_mat(data_path, args.mat_file, device)
    if args.oracle_best_f1 and ref is None:
        raise ValueError('--oracle_best_f1 requires Ref_gt; omit this flag for unlabeled training.')
    H, W = int(img_t1_raw.shape[0]), int(img_t1_raw.shape[1])

    # superpixels ref
    if args.sp_on == 't1':
        sp_ref = img_t1_raw.detach().cpu().numpy()
        sp_type = args.img_t1_type1
    elif args.sp_on == 't2':
        sp_ref = img_t2_raw.detach().cpu().numpy()
        sp_type = args.img_t2_type2
    else:
        sp_ref = 0.5 * (img_t1_raw.detach().cpu().numpy().astype(np.float32) + img_t2_raw.detach().cpu().numpy().astype(np.float32))
        sp_type = args.img_t2_type2

    labels = compute_superpixels(sp_ref, n_segments=args.N_SEG, compactness=args.Com,
                                 img_type=sp_type, sigma=args.slic_sigma, slic_zero=args.slic_zero)
    K = int(labels.max()) + 1
    edge_undir = build_edge_index(labels)
    edge_dir = edges_to_directed(edge_undir)

    # save superpixel visualization
    vis_dir = os.path.join(args.result_folder, f"sp_vis_slic{args.N_SEG}_com{args.Com:.2f}")
    ensure_dir(vis_dir)
    sp_vis_img = _prepare_slic_image(sp_ref)
    sp_vis_img = (sp_vis_img - sp_vis_img.min()) / (sp_vis_img.max() - sp_vis_img.min() + 1e-6)
    sp_bnd = mark_boundaries(sp_vis_img, labels)
    io.imsave(os.path.join(vis_dir, f"sp_boundaries_{os.path.splitext(args.mat_file)[0]}.png"),
              (sp_bnd * 255).astype(np.uint8))

    # preprocessing for model features
    # 兼容 2D/3D：如果是单通道 2D（H,W），先扩展到 (H,W,1)，统一走 process_images
    if img_t1_raw.dim() == 2:
        img_t1_raw = img_t1_raw.unsqueeze(-1)
    if img_t2_raw.dim() == 2:
        img_t2_raw = img_t2_raw.unsqueeze(-1)
    img_t1, img_t2 = process_images(img_t1_raw, img_t2_raw, args.img_t1_type1, args.img_t2_type2)


    # counts + centroid positions (for weighted Otsu + optional kNN edges)
    pos_np, counts_np, K_sp = compute_pos_counts_np(labels)
    area = counts_np.reshape(-1).astype(np.float64)
    # reduce large-superpixel dominance (improves small-object sensitivity)
    w_area = np.sqrt(np.clip(area, 1.0, None))
    w_area = w_area / (w_area.mean() + 1e-12)
    weights_u8_const = w_area[labels].astype(np.float64)

    # optionally add kNN edges on centroids
    if int(args.edge_knn_k) > 0:
        edge_knn = build_knn_edges(pos_np, int(args.edge_knn_k))
        if edge_knn.size > 0:
            edge_undir = np.unique(np.concatenate([edge_undir, edge_knn], axis=0), axis=0)
            edge_dir = edges_to_directed(edge_undir)

    # edges to torch
    edge_index_undir_t = torch.from_numpy(edge_undir).to(device=device)
    edge_index_dir_t = torch.from_numpy(edge_dir).to(device=device)

    # model + loss (CGCS-Net)
    tokens = pool_tokens(img_t1, img_t2, labels, feat_mode=args.sp_feat, device=device)
    feat_x = int(tokens.x.shape[1])
    feat_y = int(tokens.y.shape[1])

    tokens_for_w = tokens.y[:, :feat_y // 2] if args.sp_feat == 'meanstd' else tokens.y
    edge_w = compute_edge_weight(tokens_for_w, edge_index_undir_t, mode=args.edge_weight, sigma=args.edge_sigma)
    edge_w_dir = torch.cat([edge_w, edge_w], dim=0) if edge_w is not None else None

    model = CGCSNetModel(in_x=feat_x, in_y=feat_y, hidden_dim=int(args.cgcs_dim), depth=int(args.cgcs_depth),
                         drop=float(args.cgcs_drop)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = CGCSNetLoss(w_recon=float(args.cgcs_w_recon), w_co=float(args.cgcs_w_co), w_str=float(args.cgcs_w_str),
                            w_mask_l1=float(args.cgcs_w_mask_l1), w_mask_tv=float(args.cgcs_w_mask_tv),
                            w_mask_ent=float(args.cgcs_w_mask_ent), mask_warmup=int(args.cgcs_mask_warmup)).to(device)

    best_f1 = -1.0
    best_epoch = -1

    def run_eval(epoch: int):
        nonlocal best_f1, best_epoch
        model.eval()
        with torch.no_grad():
            out = model(tokens.x, tokens.y, tokens.pos,
                        edge_index_dir=edge_index_dir_t, edge_weight_dir=edge_w_dir,
                        edge_index_undir=edge_index_undir_t, edge_weight=edge_w)

            di_com = cosine_distance_cgcs(out["c_x"], out["c_y"])  # [K]
            di_mask = out["m"].clamp(0.0, 1.0)                  # [K]

            # structure DI computed in structure-branch space, and downweight suspected-change nodes
            node_w = (1.0 - di_mask.detach())
            node_w = node_w / (node_w.mean() + 1e-6)
            node_w = node_w.clamp(min=float(args.di_str_node_min), max=float(args.di_str_node_max))
            di_str = structure_diff_per_node(out["s_x"], out["s_y"], edge_index_undir_t,
                                             edge_dir=edge_index_dir_t, edge_weight=edge_w,
                                             node_weight=node_w, ctx_mix=float(args.cgcs_ctx_mix))  # [K]

            di_tok = di_com + float(args.cgcs_eta_str) * di_str
            di_mask_mix = float(np.clip(args.cgcs_di_mask_mix, 0.0, 1.0))
            if di_mask_mix > 0.0:
                di_tok = (1.0 - di_mask_mix) * di_tok + di_mask_mix * di_mask
            di_tok = di_tok * (1.0 + float(args.cgcs_kappa_mask) * di_mask)

            di_parts = [("DI_com", di_com), ("DI_str", di_str), ("DI_mask", di_mask)]

            # robust normalize with percentile clipping (on tokens)
            di_np = di_tok.detach().cpu().numpy().astype(np.float32)
            p1 = np.percentile(di_np, args.clip_p1)
            p99 = np.percentile(di_np, args.clip_p99)
            di_np = np.clip(di_np, p1, p99)
            di_np = (di_np - p1) / (p99 - p1 + 1e-6)
            di_t = torch.from_numpy(di_np).to(device)

            # graph smoothing
            di_t = graph_smooth_scores(di_t, edge_index_dir_t, edge_w, alpha=args.di_smooth_alpha, iters=args.di_smooth_iter)
            di_np2 = di_t.detach().cpu().numpy().astype(np.float32)
            di_map = (di_np2[labels] * 255.0).clip(0, 255).astype(np.uint8)

            # optional parts
            if int(args.save_di_parts) == 1:
                for name, part in di_parts:
                    pn = part.detach().cpu().numpy().astype(np.float32)
                    p1p = np.percentile(pn, args.clip_p1)
                    p99p = np.percentile(pn, args.clip_p99)
                    pn = np.clip(pn, p1p, p99p)
                    pn = (pn - p1p) / (p99p - p1p + 1e-6)
                    pu8 = (pn[labels] * 255.0).clip(0, 255).astype(np.uint8)
                    io.imsave(os.path.join(out_dir, f"{name}_iter_{epoch}.png"), pu8)

            io.imsave(os.path.join(out_dir, f"DI_iter_{epoch}.png"), di_map)

            # weights for weighted otsu: superpixel area at pixel level (constant per labels)
            weights_u8 = weights_u8_const

            cm = postprocess_cm(di_map, cm_th_mode=args.cm_th_mode, weights_u8=weights_u8,
                                tau_eval=args.tau_eval,
                                blur_sigma=args.blur_sigma, open_ks=args.open_ks, close_ks=args.close_ks,
                                open_it=args.open_it, close_it=args.close_it, min_area=args.min_area)

            io.imsave(os.path.join(out_dir, f"CM_iter_{epoch}.png"), cm)

        if ref is None:
            return None
        stats = compute_metrics(ref, cm)

        log_path = os.path.join(out_dir, 'eval.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('===============================Parameters settings==============================\n')
            f.write(f'=== epoch={epoch} || K={args.N_SEG} || compact={args.Com} || feat={args.sp_feat} || cm_th={args.cm_th_mode} ===\n')
            f.write(f"TP={stats['TP']} || TN={stats['TN']} || FP={stats['FP']} || FN={stats['FN']}\n")
            f.write(f"OA={stats['OA']}\nKappa={stats['Kappa']}\nAA={stats['AA']}\nPrecision={stats['Precision']}\nRecall={stats['Recall']}\nF1={stats['F1']}\n")

        f1_value = float(stats["F1"])

        if args.oracle_best_f1 and f1_value > best_f1:
            best_f1 = f1_value
            best_epoch = int(epoch)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'F1': f1_value},
                       os.path.join(out_dir, 'best_cgcs_net.pth'))
            with open(os.path.join(out_dir, 'best.txt'), 'w', encoding='utf-8') as f:
                f.write('===============================Best metrics==============================\n')
                f.write(f'=== epoch={epoch} || K={args.N_SEG} || compact={args.Com} || feat={args.sp_feat} || cm_th={args.cm_th_mode} ===\n')
                f.write(f"TP={stats['TP']} || TN={stats['TN']} || FP={stats['FP']} || FN={stats['FN']}\n")
                f.write(f"OA={stats['OA']}\nKappa={stats['Kappa']}\nAA={stats['AA']}\nPrecision={stats['Precision']}\nRecall={stats['Recall']}\nF1={stats['F1']}\n")

        return stats

    # train
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(tokens.x, tokens.y, tokens.pos,
                    edge_index_dir=edge_index_dir_t, edge_weight_dir=edge_w_dir,
                    edge_index_undir=edge_index_undir_t, edge_weight=edge_w)
        loss, loss_stats, di_tok, w_tok = criterion(out,
                                                    x_in=tokens.x, y_in=tokens.y,
                                                    edge_index_undir=edge_index_undir_t,
                                                    edge_weight=edge_w,
                                                    epoch=epoch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(out_dir, f'weight_epoch_{epoch}.pth'))

        print(f"[CGCS-Net] Epoch {epoch} | loss={float(loss.item()):.6f} | "
              f"recon={loss_stats['loss_recon']:.4f} co={loss_stats['loss_co']:.4f} "
              f"str={loss_stats['loss_str']:.4f} mask_l1={loss_stats['loss_mask_l1']:.4f} "
              f"mask_tv={loss_stats['loss_mask_tv']:.4f} ent={loss_stats['loss_mask_ent']:.4f} | "
              f"m_mean={loss_stats['m_mean']:.4f} di_mean={loss_stats['di_mean']:.4f}")

        run_eval(epoch)

    print(f"[CGCS-Net] best_epoch={best_epoch} best_F1={best_f1:.4f}")


if __name__ == '__main__':
    main()
