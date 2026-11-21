#!/usr/bin/env python3
"""BraTS visualisation: segmentation overlays for the 30-epoch model.

Runs the trained best_model.pth (30-epoch run, best val Dice 0.6583 at
epoch 30) on all 24 validation cases with sliding-window inference at
roi 96, picks the best, median and worst cases by mean Dice, and saves
a 3-panel axial figure for each one (FLAIR input, ground-truth overlay,
prediction overlay) plus a JSON summary of the per-case scores.

The split, transforms, model and Dice definition are identical to
train.py: the last 24 cases (sorted by file name) are validation, the
metric is the mean Dice over the three tumour classes (background
excluded), and validation uses a 96^3 sliding window, which matches the
96^3 training patches. (A diagnostic on the 10-epoch model showed
roi 96 scores higher than roi 128: mean Dice 0.5979 vs 0.5672 on 6
validation cases, so roi 96 is used throughout.)

Run in Colab from /content/brats-brain-segmentation after mounting Drive,
unzipping the project and downloading the dataset:

    !python brats-visualize.py \
        --checkpoint /content/drive/MyDrive/brats_unet/run30ep/best_model.pth \
        --data-dir data/Task01_BrainTumour \
        --output-dir /content/drive/MyDrive/brats_unet/run30ep/figures \
        --download

Figures are saved as overlay_best_<case>.png, overlay_median_<case>.png
and overlay_worst_<case>.png in the output folder, with a JSON summary
brats_visualization.json next to them.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.apps import download_and_extract
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

# Same download source as train.py (MONAI's public MSD mirror, no
# registration needed, CC BY-SA 4.0).
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_MD5 = "240a19d752f0d9e9101544901065d872"

NUM_CLASSES = 4
NUM_CHANNELS = 4  # FLAIR, T1, T1ce, T2 (in that order)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise BraTS segmentations for the best, median "
                    "and worst validation cases."
    )
    parser.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/brats_unet/run30ep/best_model.pth",
        help="Trained checkpoint (.pth), e.g. best_model.pth from the "
             "30-epoch run.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/Task01_BrainTumour",
        help="Folder with imagesTr/ and labelsTr/ "
             "(default: data/Task01_BrainTumour).",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/brats_unet/run30ep/figures",
        help="Folder for the overlay figures and the JSON summary "
             "(default: /content/drive/MyDrive/brats_unet/run30ep/figures).",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the MSD Brain Tumour data automatically if the "
             "folder is missing.",
    )
    parser.add_argument(
        "--roi-size", type=int, nargs=3, default=[96, 96, 96],
        help="Sliding-window size for inference (default: 96 96 96).",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=2,
        help="Windows per sliding-window batch (default: 2).",
    )
    return parser.parse_args()


def val_transforms():
    # Identical to train.py's val_transforms (same order, same settings).
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )


def dice_per_class(pred, label):
    """Mean Dice over the 3 tumour classes (1, 2, 3), background excluded.

    Convention: a class absent from both prediction and ground truth scores
    1.0; absent from ground truth but predicted scores 0.0.
    """
    scores = []
    for cls in (1, 2, 3):
        p = pred == cls
        g = label == cls
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(2.0 * inter / denom)
    return scores


def build_model(device):
    # Same architecture as train.py.
    return UNet(
        spatial_dims=3,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)


def run_inference(model, file_dicts, device, roi_size, sw_batch_size,
                  keep_volumes=False):
    """Run sliding-window inference on the given cases.

    Returns a list of dicts, one per case, with the case name, the mean
    Dice and the per-class Dice. When keep_volumes is True, the FLAIR
    volume (channel 0), ground-truth label and predicted label are
    included too, for the figures.
    """
    ds = Dataset(data=file_dicts, transform=val_transforms())
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    post_pred = Compose([AsDiscrete(argmax=True)])
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # MONAI 1.6 no longer puts 'image_meta_dict' in the batch, so
            # take the case name from the ordered file list (the loader
            # runs in order, batch size 1).
            name = os.path.basename(file_dicts[idx]["image"])
            inputs = batch["image"].to(device)
            outs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
            pred = post_pred(decollate_batch(outs)[0]).cpu().numpy()[0]
            label = batch["label"].cpu().numpy()[0, 0]
            scores = dice_per_class(pred, label)
            entry = {
                "case": name,
                "dice": round(float(np.mean(scores)), 4),
                "per_class": [round(s, 4) for s in scores],
            }
            if keep_volumes:
                entry["flair"] = batch["image"].cpu().numpy()[0, 0]
                entry["label"] = label
                entry["pred"] = pred
            results.append(entry)
            print(f"  {name}: dice {np.mean(scores):.4f} "
                  f"(classes {[round(s, 3) for s in scores]})")
    return results


def save_overlay_figure(flair, label, pred, case_name, dice, rank, out_path):
    """3-panel axial figure: FLAIR input, ground truth, prediction.

    The slice shown is the axial slice with the most tumour voxels in the
    ground truth. Labels: 1 oedema, 2 necrotic/non-enhancing core,
    3 enhancing tumour.
    """
    tumour_per_slice = (label > 0).sum(axis=(0, 1))
    z = int(np.argmax(tumour_per_slice))
    flair_slice = np.rot90(flair[:, :, z])
    label_slice = np.rot90(label[:, :, z])
    pred_slice = np.rot90(pred[:, :, z])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title(f"FLAIR input (slice {z})")
    axes[1].imshow(flair_slice, cmap="gray")
    axes[1].imshow(np.ma.masked_where(label_slice == 0, label_slice),
                   cmap="jet", alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title("Ground truth")
    axes[2].imshow(flair_slice, cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_slice == 0, pred_slice),
                   cmap="jet", alpha=0.6, vmin=0, vmax=3)
    axes[2].set_title(f"Prediction (Dice {dice:.3f})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"BraTS {rank} case: {case_name} (30-epoch model, roi 96)\n"
                 "Labels: 1 oedema, 2 tumour core, 3 enhancing tumour")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("figure saved:", out_path)


def main():
    args = parse_args()
    roi_size = tuple(args.roi_size)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Get the data (download it if asked, otherwise expect it on disk)
    # ------------------------------------------------------------------
    if not os.path.isdir(args.data_dir):
        if args.download:
            print(f"Downloading MSD Brain Tumour data (~5.7 GB) to {args.data_dir} ...")
            download_and_extract(
                MSD_URL,
                os.path.join(os.path.dirname(args.data_dir) or ".", "Task01_BrainTumour.tar"),
                os.path.dirname(args.data_dir) or ".",
                MSD_MD5, "md5",
            )
        else:
            raise FileNotFoundError(
                f"Data folder not found: {args.data_dir}\n"
                "Run `python download_data.py` first, or re-run with --download."
            )

    # Same split as train.py: the last 24 cases are validation.
    images = sorted(glob.glob(os.path.join(args.data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(args.data_dir, "labelsTr", "*.nii.gz")))
    val_files = [{"image": i, "label": l} for i, l in zip(images, labels)][-24:]
    print(f"validation cases: {len(val_files)}")

    # ------------------------------------------------------------------
    # 2. Load the model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = build_model(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("loaded", args.checkpoint)

    # ------------------------------------------------------------------
    # 3. Score all 24 validation cases (volumes not kept: one brain
    #    volume is ~70 MB, so keeping all 24 in RAM is wasteful)
    # ------------------------------------------------------------------
    print(f"scoring {len(val_files)} validation cases at roi {roi_size} ...")
    scored = run_inference(model, val_files, device, roi_size,
                           args.sw_batch_size, keep_volumes=False)
    mean_dice = float(np.mean([s["dice"] for s in scored]))
    print(f"mean Dice over {len(scored)} validation cases: {mean_dice:.4f}")

    ranked = sorted(scored, key=lambda d: d["dice"])
    picks = {
        "worst": ranked[0],
        "median": ranked[len(ranked) // 2],
        "best": ranked[-1],
    }
    for rank, entry in picks.items():
        print(f"  {rank}: {entry['case']} (Dice {entry['dice']:.4f})")

    # ------------------------------------------------------------------
    # 4. Re-run inference on just the 3 picked cases and save figures
    # ------------------------------------------------------------------
    by_name = {os.path.basename(v["image"]): v for v in val_files}
    pick_files = [by_name[picks[r]["case"]] for r in ("best", "median", "worst")]
    detailed = run_inference(model, pick_files, device, roi_size,
                             args.sw_batch_size, keep_volumes=True)
    detail_by_name = {d["case"]: d for d in detailed}
    figure_paths = {}
    for rank in ("best", "median", "worst"):
        entry = picks[rank]
        det = detail_by_name[entry["case"]]
        case_stub = entry["case"].replace(".nii.gz", "")
        out_path = os.path.join(args.output_dir, f"overlay_{rank}_{case_stub}.png")
        save_overlay_figure(det["flair"], det["label"], det["pred"],
                            entry["case"], entry["dice"], rank, out_path)
        figure_paths[rank] = os.path.basename(out_path)

    # ------------------------------------------------------------------
    # 5. JSON summary
    # ------------------------------------------------------------------
    summary = {
        "model": args.checkpoint,
        "roi": list(roi_size),
        "cases": len(scored),
        "mean_dice": round(mean_dice, 4),
        "per_case": scored,
        "picks": {
            rank: {"case": e["case"], "dice": e["dice"],
                   "per_class": e["per_class"], "figure": figure_paths[rank]}
            for rank, e in picks.items()
        },
    }
    json_path = os.path.join(args.output_dir, "brats_visualization.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("summary saved:", json_path)
    print("DONE")


if __name__ == "__main__":
    main()
