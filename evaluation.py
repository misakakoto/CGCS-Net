import argparse
import csv
import os
import re

import cv2
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class Evaluation:
    def __init__(self, label, pred):
        super().__init__()
        self.label = self._to_binary(label)
        self.pred = self._to_binary(pred)
        if self.label.shape != self.pred.shape:
            raise ValueError("The shape of the label and prediction must be the same.")

        size = self.label.size
        intersection = self.label * self.pred
        union = np.clip(self.label + self.pred, 0, 1)

        self.TP = int(intersection.sum())
        self.TN = int(((1 - self.label) * (1 - self.pred)).sum())
        self.FP = int(((self.pred == 1) & (self.label == 0)).sum())
        self.FN = int(((self.label == 1) & (self.pred == 0)).sum())

        self.c_num_or = int(union.sum())
        self.uc_num_or = int(size - self.c_num_or)

    def _to_binary(self, data):
        data = self._to_numpy(data).astype(np.float32)
        if data.ndim > 2:
            if data.ndim == 3:
                if data.shape[-1] == 1:
                    data = data[..., 0]
                elif data.shape[0] == 1:
                    data = data[0, ...]
                else:
                    data = data[..., 0]
            else:
                data = np.squeeze(data)
        if data.size == 0:
            return data.astype(np.uint8)
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max == data_min:
            return (data > 0).astype(np.uint8)
        if data_min < 0:
            data = (data - data_min) / (data_max - data_min)
        elif data_max > 1.0:
            data = data / data_max
        return (data > 0.5).astype(np.uint8)

    def _to_numpy(self, data):
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def matrix(self):
        return self.TP, self.TN, self.FP, self.FN

    def Classification_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        OA = (TP + TN) / (TP + TN + FP + FN)
        kappa = metrics.cohen_kappa_score(self.label.flatten(), self.pred.flatten())
        AA = (TP / (TP + FN) + TN / (TN + FP)) / 2

        return OA * 100, kappa * 100, AA * 100

    def CD_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        FA = FP / (FP + TN) if (FP + TN) != 0 else 0
        MA = FN / (FN + TP) if (FN + TP) != 0 else 0
        TE = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

        return FA * 100, MA * 100, TE * 100

    def Landslide_indicators(self):
        TP = self.TP
        FP = self.FP
        FN = self.FN

        completeness = TP / (TP + FN) if (TP + FN) != 0 else 0
        correctness = TP / (TP + FP) if (TP + FP) != 0 else 0
        quality = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        return completeness * 100, correctness * 100, quality * 100

    def IOU_indicator(self):
        c_num_and = self.TP
        c_num_or = self.c_num_or
        uc_num_and = self.TN
        uc_num_or = self.uc_num_or

        c_iou = (c_num_and / c_num_or) * 100 if c_num_or != 0 else 0
        uc_iou = (uc_num_and / uc_num_or) * 100 if uc_num_or != 0 else 0
        miou = (c_iou + uc_iou) / 2

        return miou, c_iou, uc_iou

    def ObjectExtract_indicators(self):
        TP = self.TP
        FP = self.FP
        FN = self.FN

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        epsilon = 1e-7
        f1 = (2 * precision * recall) / (precision + recall + epsilon) if (precision + recall) != 0 else 0

        return precision * 100, recall * 100, f1 * 100


CONFUSION_COLORS_BGR = {
    "TP": (255, 255, 255),
    "FP": (0, 255, 255),
    "TN": (0, 0, 0),
    "FN": (255, 255, 101),
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_gray_u8(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_heatmap(di_u8: np.ndarray) -> np.ndarray:
    pred_u8 = np.clip(di_u8, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)


def _float_map_to_u8(arr: np.ndarray) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(src, dtype=np.float32)
    valid = np.isfinite(src)
    if not np.any(valid):
        return np.zeros_like(src, dtype=np.uint8)
    lo = float(np.nanmin(src[valid]))
    hi = float(np.nanmax(src[valid]))
    if hi > lo:
        out[valid] = (src[valid] - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _confusion_color_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    pred = (pred_bin > 0).astype(np.bool_)
    gt = (gt_bin > 0).astype(np.bool_)
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tn = (~pred) & (~gt)
    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt
    out[tn] = CONFUSION_COLORS_BGR["TN"]
    out[tp] = CONFUSION_COLORS_BGR["TP"]
    out[fp] = CONFUSION_COLORS_BGR["FP"]
    out[fn] = CONFUSION_COLORS_BGR["FN"]
    return out


def _plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_dir: str, prefix: str):
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None, None, None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    aur = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    aupr = float(auc(recall, precision))

    np.savetxt(
        os.path.join(out_dir, f"{prefix}roc_curve.csv"),
        np.stack([fpr, tpr], axis=1),
        delimiter=",",
        header="fpr,tpr",
        comments="",
    )
    np.savetxt(
        os.path.join(out_dir, f"{prefix}pr_curve.csv"),
        np.stack([recall, precision], axis=1),
        delimiter=",",
        header="recall,precision",
        comments="",
    )

    if plt is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUR={aur:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}roc_curve.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, linewidth=2, label=f"AP={ap:.4f} | AUPR={aupr:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}pr_curve.png"), dpi=300)
        plt.close()

    return aur, ap, aupr


def export_eval_outputs(
    label_u8: np.ndarray,
    pred_u8: np.ndarray,
    di_u8: np.ndarray | None,
    out_dir: str,
    prefix: str = "",
    di_score: np.ndarray | None = None,
) -> dict:
    _ensure_dir(out_dir)

    conf = _confusion_color_map(pred_u8, label_u8)
    conf_path = os.path.join(out_dir, f"{prefix}confusion.png")
    cv2.imwrite(conf_path, conf)

    roc_auc = None
    ap = None
    aupr = None
    heat_path = None
    score_source = None
    score_unique = None
    score_map = None

    if di_score is not None:
        score_map = np.asarray(di_score, dtype=np.float32)
        if score_map.shape != label_u8.shape:
            raise ValueError("di_score and label must have the same shape.")
        score_source = "float_score_map"
    elif di_u8 is not None:
        score_map = di_u8.astype(np.float32) / 255.0
        score_source = "uint8_di_div255"

    if di_u8 is not None:
        heat = _to_heatmap(di_u8)
        heat_path = os.path.join(out_dir, f"{prefix}di_heatmap.png")
        cv2.imwrite(heat_path, heat)
    elif score_map is not None:
        heat = _to_heatmap(_float_map_to_u8(score_map))
        heat_path = os.path.join(out_dir, f"{prefix}di_heatmap.png")
        cv2.imwrite(heat_path, heat)

    if score_map is not None:
        y_true = (label_u8 > 0).astype(np.uint8).reshape(-1)
        y_score = score_map.reshape(-1)
        valid = np.isfinite(y_score)
        y_true = y_true[valid]
        y_score = y_score[valid]
        score_unique = int(np.unique(y_score).size)
        roc_auc, ap, aupr = _plot_roc_pr(y_true, y_score, out_dir, prefix)

    indicators = Evaluation(label_u8, pred_u8)
    OA, kappa, AA = indicators.Classification_indicators()
    Precision, Recall, F1 = indicators.ObjectExtract_indicators()
    TP, TN, FP, FN = indicators.matrix()

    metrics_path = os.path.join(out_dir, f"{prefix}metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"TP={TP} || TN={TN} || FP={FP} || FN={FN}\n")
        f.write(f"OA={OA}\nKappa={kappa}\nAA={AA}\n")
        f.write(f"Precision={Precision}\nRecall={Recall}\nF1={F1}\n")
        if score_source is not None:
            f.write(f"AUC_SCORE_SOURCE={score_source}\n")
        if score_unique is not None:
            f.write(f"AUC_SCORE_UNIQUE={score_unique}\n")
            if score_unique <= 2:
                f.write("AUC_SCORE_WARNING=score is near-binary; use continuous DI/probability for ROC/PR.\n")
        if roc_auc is not None and ap is not None and aupr is not None:
            f.write(f"ROC_AUC={roc_auc}\nAUR={roc_auc}\nAP={ap}\nAUPR={aupr}\nAUP={aupr}\n")
        f.write("Color map (confusion): TP=white, FP=yellow, FN=(255,255,101), TN=black\n")

    return {
        "confusion_png": conf_path,
        "heatmap_png": heat_path,
        "metrics_txt": metrics_path,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "OA": OA,
        "Kappa": kappa,
        "AA": AA,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "roc_auc": roc_auc,
        "ap": ap,
        "aupr": aupr,
        "auc_score_source": score_source,
        "auc_score_unique": score_unique,
    }


def _normalize_dataset_tag(tag: str) -> str:
    raw = str(tag).strip()
    if not raw:
        raise ValueError("dataset tag is empty.")
    if raw.startswith("dataset#"):
        return raw
    if raw.startswith("d") and raw[1:].isdigit():
        return f"dataset#{raw[1:]}"
    if raw.isdigit():
        return f"dataset#{raw}"
    return raw


def _parse_best_epoch(best_txt_path: str) -> int:
    with open(best_txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"epoch\s*=\s*(\d+)", content)
    if match is None:
        raise ValueError(f"Cannot parse best epoch from {best_txt_path}")
    return int(match.group(1))


def _discover_dataset_tags(result_root: str, dataset_filters: list[str] | None = None) -> list[str]:
    if dataset_filters:
        return [_normalize_dataset_tag(item) for item in dataset_filters]
    if not os.path.isdir(result_root):
        raise FileNotFoundError(f"Result root not found: {result_root}")
    tags = []
    for name in sorted(os.listdir(result_root)):
        path = os.path.join(result_root, name)
        if os.path.isdir(path) and name.startswith("dataset#"):
            tags.append(name)
    return tags


def _build_eval_job(
    dataset_tag: str,
    result_root: str,
    label_root: str,
    epoch: int | None = None,
    result_subdir: str = "cgcs_net_results",
) -> dict:
    dataset_tag = _normalize_dataset_tag(dataset_tag)
    result_dir = os.path.join(result_root, dataset_tag, result_subdir)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    best_txt_path = os.path.join(result_dir, "best.txt")
    if epoch is None:
        if not os.path.isfile(best_txt_path):
            raise FileNotFoundError(f"best.txt not found: {best_txt_path}")
        epoch = _parse_best_epoch(best_txt_path)

    pred_path = os.path.join(result_dir, f"CM_iter_{epoch}.png")
    di_path = os.path.join(result_dir, f"DI_iter_{epoch}.png")
    label_path = os.path.join(label_root, dataset_tag, "Ref_gt.png")

    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not os.path.isfile(di_path):
        di_path = None

    return {
        "dataset_tag": dataset_tag,
        "epoch": int(epoch),
        "result_dir": result_dir,
        "pred_path": pred_path,
        "label_path": label_path,
        "di_path": di_path,
        "best_txt_path": best_txt_path if os.path.isfile(best_txt_path) else None,
    }


def _run_eval_job(job: dict, out_root: str, prefix: str = "") -> dict:
    dataset_tag = job["dataset_tag"]
    pred = _read_gray_u8(job["pred_path"])
    label = _read_gray_u8(job["label_path"])
    di = _read_gray_u8(job["di_path"]) if job["di_path"] else None
    out_dir = os.path.join(out_root, dataset_tag)
    result = export_eval_outputs(label, pred, di, out_dir, prefix)
    result.update(
        {
            "dataset": dataset_tag,
            "epoch": job["epoch"],
            "pred_path": job["pred_path"],
            "label_path": job["label_path"],
            "di_path": job["di_path"] or "",
            "result_dir": job["result_dir"],
            "best_txt_path": job["best_txt_path"] or "",
            "status": "ok",
            "error": "",
        }
    )
    return result


def _write_summary(rows: list[dict], out_root: str, name: str = "summary.csv") -> str:
    _ensure_dir(out_root)
    out_path = os.path.join(out_root, name)
    fieldnames = [
        "dataset",
        "status",
        "epoch",
        "F1",
        "Precision",
        "Recall",
        "OA",
        "Kappa",
        "AA",
        "TP",
        "TN",
        "FP",
        "FN",
        "roc_auc",
        "ap",
        "aupr",
        "pred_path",
        "label_path",
        "di_path",
        "metrics_txt",
        "confusion_png",
        "heatmap_png",
        "result_dir",
        "best_txt_path",
        "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export evaluation artifacts for one or more datasets without editing paths."
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default=r"results\save",
        help="Root folder containing dataset result subfolders.",
    )
    parser.add_argument(
        "--label-root",
        type=str,
        default=r"data\LPEM\png",
        help="Root folder containing ground-truth labels.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"eval_out",
        help="Root folder for exported evaluation outputs.",
    )
    parser.add_argument(
        "--result-subdir",
        type=str,
        default="cgcs_net_results",
        help="Result subfolder inside each dataset folder.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset tag like dataset#15, 15, or d15. Can be used multiple times.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Override epoch for a single dataset. Default reads best.txt.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.csv",
        help="CSV filename written under out-root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when any dataset cannot be evaluated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_tags = _discover_dataset_tags(args.result_root, args.dataset)
    if not dataset_tags:
        raise FileNotFoundError(f"No dataset folders found under {args.result_root}")
    if args.epoch is not None and len(dataset_tags) != 1:
        raise ValueError("--epoch can only be used with exactly one dataset.")

    rows = []
    ok_count = 0
    fail_count = 0

    for dataset_tag in dataset_tags:
        try:
            job = _build_eval_job(
                dataset_tag=dataset_tag,
                result_root=args.result_root,
                label_root=args.label_root,
                epoch=args.epoch,
                result_subdir=args.result_subdir,
            )
            row = _run_eval_job(job, args.out_root)
            rows.append(row)
            ok_count += 1
            print(f"[OK] {dataset_tag} epoch={row['epoch']} F1={row['F1']:.4f}")
        except Exception as exc:
            fail_count += 1
            rows.append(
                {
                    "dataset": _normalize_dataset_tag(dataset_tag),
                    "status": "error",
                    "epoch": args.epoch if args.epoch is not None else "",
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {dataset_tag}: {exc}")
            if args.strict:
                raise

    summary_path = _write_summary(rows, args.out_root, args.summary_name)
    print(f"[DONE] success={ok_count} failed={fail_count} summary={summary_path}")


if __name__ == "__main__":
    main()
