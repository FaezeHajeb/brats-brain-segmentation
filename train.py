#!/usr/bin/env python3
"""
3D brain tumour segmentation from multi-modal MRI with MONAI + UNet.

This script trains a 3D UNet to segment brain tumours (gliomas) in
multi-modal MRI, using the Medical Segmentation Decathlon (MSD) Task01
Brain Tumour dataset -- BraTS data packaged by the Decathlon: the same
four MRI sequences and the same expert labels as the BraTS challenge.

Each case is one 4-channel MRI volume (FLAIR, T1, T1ce, T2) and one
expert mask with four classes:

    0  background
    1  peritumoural oedema
    2  necrotic and non-enhancing tumour core
    3  enhancing tumour

Note: these are the MSD labels, which differ from the original BraTS
labels. BraTS uses 1 = necrotic core, 2 = oedema, 4 = enhancing tumour
(there is no label 3); the Decathlon remapped them to the continuous
0-3 scheme above (the same remapping nnU-Net uses). The values were
verified in the dataset's own dataset.json, which lists
1 = edema, 2 = non-enhancing tumor, 3 = enhancing tumour -- so no
remapping is needed in this script.

Because the labels are already 0-3, the network has 4 output channels:
background, oedema, core, enhancing tumour. The BraTS challenge usually
reports Dice per region instead (whole tumour = labels 1+2+3, tumour
core = labels 2+3, enhancing tumour = label 3); this simpler 4-class
formulation is easier to learn from, and the mean Dice over the three
tumour classes is reported here. See README.md for the difference.

What you will learn from this script
--------------------------------------
1. How to load multi-channel 3D MRI (NIfTI files) with MONAI.
2. How MRI preprocessing differs from CT: there is no fixed intensity
   scale like Hounsfield units, so each channel is standardised
   (mean 0, standard deviation 1) over the brain voxels instead.
3. How to train a 3D UNet with Dice loss on a multi-class problem.
4. How to validate with sliding-window inference (needed because a
   whole brain volume does not fit in GPU memory at once).
5. How to save the best model and log the training history.

Quick start
-----------
    # 1. Download the data (about 5.7 GB, see download_data.py)
    python download_data.py

    # 2. Smoke test: 1 epoch on 2 cases, just to prove everything runs
    python train.py --quick

    # 3. Real training (default: 30 epochs, needs a GPU - see README.md)
    python train.py --data-dir data/Task01_BrainTumour --output-dir runs/brats_unet

    # 3b. Resume an interrupted run from its best checkpoint. Example:
    #     the 30-epoch run was cut off by Colab after epoch 16, with the
    #     best model (epoch 15) saved in the output folder. This continues
    #     with epochs 17-30 and appends to the existing metrics.csv:
    python train.py --data-dir data/Task01_BrainTumour --output-dir runs/brats_unet \
        --resume runs/brats_unet/best_model.pth --start-epoch 16

    # 4. Segment a new MRI volume with the trained model
    python predict.py --checkpoint runs/brats_unet/best_model.pth \
        --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
"""

import argparse
import csv
import glob
import json
import os
import time

import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
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
    RandCropByPosNegLabeld,
    Spacingd,
)
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# Dataset download (MONAI's public mirror of the MSD Brain Tumour data).
# No registration is needed for this link; the data is released under
# the CC BY-SA 4.0 licence (see README.md).
# ---------------------------------------------------------------------------
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_MD5 = "240a19d752f0d9e9101544901065d872"

# MSD Task01 labels (verified in dataset.json): 0 background,
# 1 peritumoural oedema, 2 necrotic/non-enhancing core, 3 enhancing
# tumour. No remapping is needed -- the labels are already 0-3.
NUM_CLASSES = 4
NUM_CHANNELS = 4  # FLAIR, T1, T1ce, T2 (in that order)


def parse_args():
    """Read the command-line options (run `python train.py --help` to see them)."""
    parser = argparse.ArgumentParser(
        description="Train a 3D UNet for brain tumour segmentation on MSD Task01."
    )
    parser.add_argument(
        "--data-dir",
        default="data/Task01_BrainTumour",
        help="Folder with imagesTr/ and labelsTr/ (default: data/Task01_BrainTumour).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/brats_unet",
        help="Folder for checkpoints, logs and plots (default: runs/brats_unet).",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--val-interval", type=int, default=5,
        help="Validate every N epochs (default: 5).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for the Adam optimiser (default: 0.0001).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Cases per training step (default: 2).",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the MSD Brain Tumour data automatically if the folder is missing.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test: 1 epoch on 2 training cases + 1 validation case.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint (.pth) to continue from, e.g. a saved best_model.pth. "
             "The model weights are loaded; the optimiser restarts fresh, "
             "because checkpoints store weights only, not optimiser state.",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0,
        help="Epochs already completed (default: 0). With --resume, training "
             "continues with epoch start-epoch + 1 up to --epochs, and new "
             "rows are APPENDED to the existing metrics.csv instead of "
             "overwriting it. The best Dice so far is recovered from that "
             "file, so best_model.pth is only replaced by a better model.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed, so results are reproducible (default: 0).",
    )
    return parser.parse_args()


def get_transforms(patch_size=(96, 96, 96), num_samples=2):
    """Build the preprocessing pipelines for training and validation.

    A "transform" is one small step applied to every image. MONAI chains
    them with Compose, and the trailing "d" means the transform works on
    dictionaries, e.g. {"image": ..., "label": ...}.

    Training transforms (in order):
      LoadImaged          read the NIfTI files (image + label)
      EnsureChannelFirstd make the shapes (4, x, y, z) for the image
                          (FLAIR, T1, T1ce, T2 -- in that order) and
                          (1, x, y, z) for the label
      NormalizeIntensityd standardise each MRI channel separately:
                          subtract its mean and divide by its standard
                          deviation, computed over the non-zero voxels
                          (the brain). MRI has no fixed intensity scale
                          like CT's Hounsfield units, so this per-scan
                          standardisation is what makes scans from
                          different machines comparable.
      CropForegroundd     cut away the empty (zero) borders around the head
      Orientationd        rotate every volume into the same RAS orientation
      Spacingd            resample every volume to 1 x 1 x 1 mm voxels.
                          BraTS data is already 1 mm isotropic, so this
                          is a safety net, not a change.
      RandCropByPosNegLabeld cut random 96x96x96 patches; half of them are
                          centred on tumour (positive) and half on
                          background (negative), so the network does not
                          only see healthy brain

    Validation transforms are the same, minus the random crop: validation
    runs on the whole volume with sliding-window inference.
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),  # smooth for image, exact for label
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1, neg=1,          # 1 positive patch per 1 negative patch
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    return train_transforms, val_transforms


def main():
    args = parse_args()
    set_determinism(seed=args.seed)  # same seed -> same random crops -> same result
    print_config()  # prints MONAI / PyTorch / GPU versions (useful for debugging)
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

    # Pair every training image with its label (same file name in both folders).
    train_images = sorted(glob.glob(os.path.join(args.data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": img, "label": lab}
        for img, lab in zip(train_images, train_labels)
    ]
    # Split: the last 24 cases are validation (460 train / 24 val).
    train_files, val_files = data_dicts[:-24], data_dicts[-24:]
    if args.quick:
        # Smoke test: tiny subset so one epoch finishes in ~1 minute on CPU.
        train_files, val_files = train_files[:2], val_files[:1]
        args.epochs = 1
        print("QUICK MODE: 1 epoch, 2 training cases, 1 validation case.")
    print(f"Training cases: {len(train_files)}, validation cases: {len(val_files)}")

    # ------------------------------------------------------------------
    # 2. Datasets and loaders
    #    CacheDataset loads each volume once, applies the deterministic
    #    transforms (everything except the random crop) and keeps the
    #    result in RAM, so later epochs are much faster than a plain
    #    Dataset. But 460 four-channel brain volumes do NOT fit in RAM
    #    (about 70 MB each after cropping), so only 10% are cached and
    #    the rest are loaded on demand. The 24 validation cases are
    #    small enough to cache fully.
    # ------------------------------------------------------------------
    train_transforms, val_transforms = get_transforms()
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0 if args.quick else 0.1,
        num_workers=0 if args.quick else 4,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=0 if args.quick else 4,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.quick else 4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0 if args.quick else 4)

    # ------------------------------------------------------------------
    # 3. Model, loss, optimiser, metric
    #    UNet: the classic encoder-decoder for medical segmentation.
    #      - spatial_dims=3      : 3D volumes (not 2D slices)
    #      - in_channels=4       : FLAIR, T1, T1ce, T2 (one per channel)
    #      - out_channels=4      : background, oedema, tumour core,
    #                              enhancing tumour
    #      - channels=(16,32,64,128,256): feature maps per level
    #      - strides=(2,2,2,2)   : downsampling between levels
    #      - num_res_units=2     : residual blocks (helps training)
    #    DiceLoss: 1 - Dice score. It directly optimises the overlap
    #      between prediction and ground truth, which is what we care
    #      about (the tumour is small, so plain accuracy would be
    #      misleading: predicting "all background" is already ~99%
    #      accurate but useless).
    #    DiceMetric: the evaluation score, background class excluded,
    #      averaged over the three tumour classes.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(
        spatial_dims=3,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    if args.resume:
        # Resume: load the saved weights into the model. The optimiser
        # below still starts fresh (checkpoints store weights only), which
        # is fine for a warm start at the same learning rate.
        print(f"Resuming from checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict)
        print("checkpoint weights loaded")
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Post-processing: turn the network's 4-channel output into a label map.
    #   argmax picks the winning class per voxel; to_onehot=4 keeps the
    #   shape (4, x, y, z) so the metric can compare it with the label.
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

    # TensorBoard is optional: if it is installed we log there too,
    # otherwise we only write metrics.csv (which is always written).
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    except Exception as exc:  # TensorBoard not installed -> CSV only
        print(f"TensorBoard unavailable ({exc}); logging to CSV only.")
        writer = None

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    max_epochs = args.epochs
    best_metric = -1.0
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    # Validation roi: 96 matches the 96^3 training patches. Diagnostic 19 Nov 2026
    # (6 val cases, 10-epoch model): mean Dice 0.5979 at roi 96 vs 0.5672 at roi 128.
    roi_size = (96, 96, 96)
    sw_batch_size = 1 if args.quick else 2

    start_epoch = args.start_epoch
    if start_epoch > 0 and not args.resume:
        raise ValueError(
            "--start-epoch needs --resume: there are no weights to continue from."
        )
    if start_epoch >= max_epochs:
        raise ValueError(
            f"--start-epoch ({start_epoch}) must be smaller than --epochs ({max_epochs})."
        )

    # Resume bookkeeping: if epochs were already completed, recover the best
    # validation Dice from the existing metrics.csv and APPEND new rows to it
    # instead of overwriting it. The checkpoint is the best model so far, so
    # any epochs after it (e.g. epoch 16 after an epoch-15 checkpoint) are not
    # replayed; their logged metrics stay as they are.
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_mode, write_header = "w", True
    if start_epoch > 0 and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    val_dice = float(row["val_dice"]) if row["val_dice"] else None
                except (ValueError, KeyError):
                    val_dice = None
                if val_dice is not None and val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = int(row["epoch"])
        print(
            f"resuming after epoch {start_epoch}: best val Dice so far "
            f"{best_metric:.4f} at epoch {best_metric_epoch}"
        )
        csv_mode, write_header = "a", False
    elif start_epoch > 0:
        print(
            f"warning: no metrics.csv found in {args.output_dir}; "
            "best Dice tracking restarts from scratch"
        )

    with open(csv_path, csv_mode, newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["epoch", "train_loss", "val_dice", "best_dice"])
        start_time = time.time()

        for epoch in range(start_epoch, max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()  # training mode: dropout/batchnorm behave accordingly
            epoch_loss = 0.0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                optimizer.zero_grad()          # clear gradients from last step
                outputs = model(inputs)        # forward pass
                loss = loss_function(outputs, labels)
                loss.backward()                # backward pass
                optimizer.step()               # update the weights
                epoch_loss += loss.item()
                print(f"  step {step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if writer is not None:
                writer.add_scalar("train/loss", epoch_loss, epoch + 1)

            # ---- Validation every val_interval epochs -----------------
            # A whole brain volume is too big for the GPU, so validation
            # uses sliding_window_inference: the volume is cut into
            # overlapping 128x128x128 windows, each is segmented, and
            # the results are stitched back together.
            val_dice = None
            if (epoch + 1) % args.val_interval == 0 or epoch == max_epochs - 1:
                model.eval()  # evaluation mode (no dropout, frozen batchnorm)
                with torch.no_grad():  # no gradients needed -> saves memory
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device)
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model
                        )
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        dice_metric(y_pred=val_outputs, y=val_labels)
                    val_dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                metric_values.append(val_dice)
                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_dice:.4f}\n"
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                if writer is not None:
                    writer.add_scalar("val/dice", val_dice, epoch + 1)

            csv_writer.writerow(
                [epoch + 1, f"{epoch_loss:.4f}",
                 f"{val_dice:.4f}" if val_dice is not None else "",
                 f"{best_metric:.4f}" if best_metric > 0 else ""]
            )
            csv_file.flush()

    # Save the final model too (even if it is not the best one).
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    if writer is not None:
        writer.close()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print(f"total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"results saved in: {args.output_dir}")

    # Remember the run settings, so the run can be reproduced later.
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
