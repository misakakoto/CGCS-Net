import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from skimage import io

import train as core
from evaluation import export_eval_outputs
from model_cgcs_net import CGCSNetModel, cosine_distance as cosine_distance_cgcs
from preprocess import process_images


def build_test_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test script for CGCS-Net using saved hparams + checkpoint from main_A.py"
    )
    p.add_argument("--result_folder", type=str, default=None,
                   help="Run result folder, e.g. results/dataset#3")
    p.add_argument("--hparams_json", type=str, default=None,
                   help="Path to run_hparams_latest.json or archived run_hparams_*.json")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (.pth). If not set, auto-select by --checkpoint_mode")
    p.add_argument("--checkpoint_mode", type=str, default="latest",
                   choices=["best", "latest", "epoch"],
                   help="How to auto-select checkpoint under cgcs_net_results")
    p.add_argument("--epoch", type=int, default=None,
                   help="Used when --checkpoint_mode=epoch (loads weight_epoch_{epoch}.pth)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to save test outputs. Default: <cgcs_net_results>/test_<ckpt_name>")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--data_dir", type=str, default=None, help="Override MAT data directory")
    return p


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_hparams(hparams_json: str) -> tuple[dict, dict]:
    with open(hparams_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    args = payload.get("args", payload)
    if not isinstance(args, dict):
        raise ValueError(f"Invalid hparams format: {hparams_json}")
    return args, payload


def _merge_saved_args(saved_args: dict) -> SimpleNamespace:
    defaults = vars(core.build_argparser().parse_args([])).copy()
    defaults.update(saved_args)
    return SimpleNamespace(**defaults)


def _parse_weight_epoch_name(filename: str) -> int | None:
    name = os.path.basename(filename)
    if not name.startswith("weight_epoch_") or not name.endswith(".pth"):
        return None
    body = name[len("weight_epoch_"):-len(".pth")]
    if not body.isdigit():
        return None
    return int(body)


def _find_latest_weight(out_dir: str) -> str:
    cands = []
    for name in os.listdir(out_dir):
        ep = _parse_weight_epoch_name(name)
        if ep is not None:
            cands.append((ep, os.path.join(out_dir, name)))
    if not cands:
        raise FileNotFoundError(f"No weight_epoch_*.pth found under: {out_dir}")
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _resolve_checkpoint(cli_args: argparse.Namespace, cgcs_out_dir: str) -> str:
    if cli_args.checkpoint:
        return cli_args.checkpoint
    mode = cli_args.checkpoint_mode
    if mode == "best":
        best = os.path.join(cgcs_out_dir, "best_cgcs_net.pth")
        if os.path.exists(best):
            return best
        return _find_latest_weight(cgcs_out_dir)
    if mode == "latest":
        return _find_latest_weight(cgcs_out_dir)
    if cli_args.epoch is None:
        raise ValueError("--epoch is required when --checkpoint_mode=epoch")
    fp = os.path.join(cgcs_out_dir, f"weight_epoch_{int(cli_args.epoch)}.pth")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return fp


def _make_sp_reference(args: SimpleNamespace, img_t1_raw: torch.Tensor, img_t2_raw: torch.Tensor):
    if args.sp_on == "t1":
        return img_t1_raw.detach().cpu().numpy(), args.img_t1_type1
    if args.sp_on == "t2":
        return img_t2_raw.detach().cpu().numpy(), args.img_t2_type2
    sp_ref = 0.5 * (
        img_t1_raw.detach().cpu().numpy().astype(np.float32) +
        img_t2_raw.detach().cpu().numpy().astype(np.float32)
    )
    return sp_ref, args.img_t2_type2


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cli_args = build_test_parser().parse_args()

    if cli_args.hparams_json is None:
        if cli_args.result_folder is None:
            raise ValueError("Provide --hparams_json or --result_folder.")
        cli_args.hparams_json = os.path.join(
            cli_args.result_folder, "cgcs_net_results", "run_hparams_latest.json"
        )
    if not os.path.exists(cli_args.hparams_json):
        raise FileNotFoundError(cli_args.hparams_json)

    saved_args_dict, payload = _load_hparams(cli_args.hparams_json)
    args = _merge_saved_args(saved_args_dict)
    if cli_args.data_dir is not None:
        args.root_dir = os.path.abspath(cli_args.data_dir)
        args.load_data_dir = '.'

    if cli_args.result_folder is not None:
        args.result_folder = cli_args.result_folder
    if args.result_folder is None:
        name = os.path.splitext(os.path.basename(args.mat_file))[0]
        args.result_folder = os.path.join("results", name)

    cgcs_out_dir = os.path.join(args.result_folder, "cgcs_net_results")
    checkpoint_path = _resolve_checkpoint(cli_args, cgcs_out_dir)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    device = _choose_device(cli_args.device)

    run_seed = int(getattr(args, "seed", 42))
    _set_seed(run_seed)

    print(f"[TEST] device={device}")
    print(f"[TEST] hparams={cli_args.hparams_json}")
    if isinstance(payload, dict):
        print(f"[TEST] run_id={payload.get('run_id', 'N/A')}")
    print(f"[TEST] checkpoint={checkpoint_path}")

    data_path = os.path.join(args.root_dir, args.load_data_dir)
    img_t1_raw, img_t2_raw, ref = core.load_images_mat(data_path, args.mat_file, device)

    sp_ref, sp_type = _make_sp_reference(args, img_t1_raw, img_t2_raw)
    labels = core.compute_superpixels(
        sp_ref,
        n_segments=int(args.N_SEG),
        compactness=float(args.Com),
        img_type=sp_type,
        sigma=float(args.slic_sigma),
        slic_zero=bool(args.slic_zero),
    )

    edge_undir = core.build_edge_index(labels)
    edge_dir = core.edges_to_directed(edge_undir)

    if img_t1_raw.dim() == 2:
        img_t1_raw = img_t1_raw.unsqueeze(-1)
    if img_t2_raw.dim() == 2:
        img_t2_raw = img_t2_raw.unsqueeze(-1)
    img_t1, img_t2 = process_images(img_t1_raw, img_t2_raw, args.img_t1_type1, args.img_t2_type2)

    pos_np, counts_np, _ = core.compute_pos_counts_np(labels)
    area = counts_np.reshape(-1).astype(np.float64)
    w_area = np.sqrt(np.clip(area, 1.0, None))
    w_area = w_area / (w_area.mean() + 1e-12)
    weights_u8_const = w_area[labels].astype(np.float64)

    if int(args.edge_knn_k) > 0:
        edge_knn = core.build_knn_edges(pos_np, int(args.edge_knn_k))
        if edge_knn.size > 0:
            edge_undir = np.unique(np.concatenate([edge_undir, edge_knn], axis=0), axis=0)
            edge_dir = core.edges_to_directed(edge_undir)

    edge_index_undir_t = torch.from_numpy(edge_undir).to(device=device)
    edge_index_dir_t = torch.from_numpy(edge_dir).to(device=device)

    tokens = core.pool_tokens(img_t1, img_t2, labels, feat_mode=args.sp_feat, device=device)
    feat_x = int(tokens.x.shape[1])
    feat_y = int(tokens.y.shape[1])
    tokens_for_w = tokens.y[:, :feat_y // 2] if args.sp_feat == "meanstd" else tokens.y
    edge_w = core.compute_edge_weight(
        tokens_for_w, edge_index_undir_t, mode=args.edge_weight, sigma=float(args.edge_sigma)
    )
    edge_w_dir = torch.cat([edge_w, edge_w], dim=0) if edge_w is not None else None

    model = CGCSNetModel(
        in_x=feat_x,
        in_y=feat_y,
        hidden_dim=int(args.cgcs_dim),
        depth=int(args.cgcs_depth),
        drop=float(args.cgcs_drop),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        out = model(
            tokens.x,
            tokens.y,
            tokens.pos,
            edge_index_dir=edge_index_dir_t,
            edge_weight_dir=edge_w_dir,
            edge_index_undir=edge_index_undir_t,
            edge_weight=edge_w,
        )

        di_com = cosine_distance_cgcs(out["c_x"], out["c_y"])
        di_mask = out["m"].clamp(0.0, 1.0)

        node_w = (1.0 - di_mask.detach())
        node_w = node_w / (node_w.mean() + 1e-6)
        node_w = node_w.clamp(min=float(args.di_str_node_min), max=float(args.di_str_node_max))

        di_str = core.structure_diff_per_node(
            out["s_x"],
            out["s_y"],
            edge_index_undir_t,
            edge_dir=edge_index_dir_t,
            edge_weight=edge_w,
            node_weight=node_w,
            ctx_mix=float(args.cgcs_ctx_mix),
        )

        di_tok = di_com + float(args.cgcs_eta_str) * di_str
        di_mask_mix = float(np.clip(args.cgcs_di_mask_mix, 0.0, 1.0))
        if di_mask_mix > 0.0:
            di_tok = (1.0 - di_mask_mix) * di_tok + di_mask_mix * di_mask
        di_tok = di_tok * (1.0 + float(args.cgcs_kappa_mask) * di_mask)

        di_np = di_tok.detach().cpu().numpy().astype(np.float32)
        p1 = np.percentile(di_np, float(args.clip_p1))
        p99 = np.percentile(di_np, float(args.clip_p99))
        di_np = np.clip(di_np, p1, p99)
        di_np = (di_np - p1) / (p99 - p1 + 1e-6)
        di_t = torch.from_numpy(di_np).to(device)

        di_t = core.graph_smooth_scores(
            di_t,
            edge_index_dir_t,
            edge_w,
            alpha=float(args.di_smooth_alpha),
            iters=int(args.di_smooth_iter),
        )

        di_np2 = di_t.detach().cpu().numpy().astype(np.float32)
        di_score_map = di_np2[labels].astype(np.float32)
        di_map = (di_score_map * 255.0).clip(0, 255).astype(np.uint8)

        cm = core.postprocess_cm(
            di_map,
            cm_th_mode=args.cm_th_mode,
            weights_u8=weights_u8_const,
            tau_eval=float(args.tau_eval),
            blur_sigma=float(args.blur_sigma),
            open_ks=int(args.open_ks),
            close_ks=int(args.close_ks),
            open_it=int(args.open_it),
            close_it=int(args.close_it),
            min_area=int(args.min_area),
        )

    if cli_args.out_dir is None:
        ckpt_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
        out_dir = os.path.join(cgcs_out_dir, f"test_{ckpt_tag}")
    else:
        out_dir = cli_args.out_dir
    core.ensure_dir(out_dir)

    io.imsave(os.path.join(out_dir, f"{cli_args.prefix}DI_test.png"), di_map)
    io.imsave(os.path.join(out_dir, f"{cli_args.prefix}CM_test.png"), cm)
    np.save(os.path.join(out_dir, f"{cli_args.prefix}DI_test_float.npy"), di_score_map.astype(np.float32))

    if ref is None:
        print(f"[TEST] DI and CM saved to {out_dir}; no Ref_gt supplied, skipping evaluation.")
        return
    stats = core.compute_metrics(ref, cm)
    metrics_dump = {
        "hparams_json": os.path.abspath(cli_args.hparams_json),
        "checkpoint": os.path.abspath(checkpoint_path),
        "dataset": os.path.join(args.root_dir, args.load_data_dir, args.mat_file),
        "metrics": stats,
    }
    with open(os.path.join(out_dir, f"{cli_args.prefix}test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dump, f, ensure_ascii=False, indent=2)

    export_eval_outputs(ref, cm, di_map, out_dir, prefix=cli_args.prefix, di_score=di_score_map)

    print("[TEST] Done.")
    print(f"[TEST] outputs={out_dir}")
    print(
        "[TEST] OA={OA:.4f} Kappa={Kappa:.4f} AA={AA:.4f} Precision={Precision:.4f} "
        "Recall={Recall:.4f} F1={F1:.4f}".format(**stats)
    )


if __name__ == "__main__":
    main()
