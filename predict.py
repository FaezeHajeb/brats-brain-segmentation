#!/usr/bin/env python3
"""
Segment one multi-modal brain MRI volume with a trained tumour model
and save the result.

The prediction is written back into the ORIGINAL image space (same size,
voxel spacing and orientation as the input file), so you can open the
input MRI and the predicted mask side by side in 3D Slicer or ITK-SNAP
and compare them. This is done with MONAI's Invertd transform, which
undoes the resampling/orientation steps applied before inference.

The predicted mask uses the MSD Task01 labels (verified in the
dataset's dataset.json):

    0 = background
    1 = peritumoural oedema
    2 = necrotic and non-enhancing tumour core
    3 = enhancing tumour

Note: these differ from the original BraTS labels (1 = necrotic core,
2 = oedema, 4 = enhancing tumour); the Decathlon remapped them to
the continuous 0-3 scheme above.

Usage
-----
    python predict.py --checkpoint runs/brats_unet/best_model.pth \
        --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz

    # output: outputs/BRATS_001_seg.nii.gz  (0-3, see the labels above)

    # segment every case in a folder:
    python predict.py --checkpoint runs/brats_unet/best_model.pth \
        --input data/Task01_BrainTumour/imagesTs
"""

import argparse
import glob
import os

import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SaveImaged,
    Spacingd,
)
from monai.utils import convert_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment brain MRI volumes with a trained tumour UNet."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a trained model, e.g. runs/brats_unet/best_model.pth",
    )
    parser.add_argument(
        "--input", required=True,
        help="One NIfTI file (.nii.gz) or a folder of NIfTI files.",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Folder for the predicted masks (default: outputs).",
    )
    parser.add_argument(
        # Default 96: matches the 96^3 training patches (diagnostic 19 Nov 2026:
        # mean Dice 0.5979 at roi 96 vs 0.5672 at roi 128 on 6 val cases).
        "--roi-size", type=int, nargs=3, default=[96, 96, 96],
        help="Sliding-window size, e.g. --roi-size 96 96 96 (default).",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=2,
        help="Windows processed at once (default: 2; lower it if you run "
        "out of GPU memory).",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda or cpu (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def build_model(checkpoint, device):
    """Create the same UNet as in train.py and load the trained weights."""
    model = UNet(
        spatial_dims=3,
        in_channels=4,   # FLAIR, T1, T1ce, T2
        out_channels=4,  # background, oedema, core, enhancing tumour
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    # weights_only=True is safe here: a checkpoint is just a dict of tensors.
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Same preprocessing as validation in train.py, but WITHOUT the label:
    #   load -> channel-first -> standardise each channel (mean 0, std 1
    #   over the brain voxels) -> RAS orientation -> resample to 1 mm.
    infer_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ]
    )

    # After inference, Invertd undoes the transforms above, so the mask
    # comes back to the original voxel grid; SaveImaged writes it as NIfTI.
    # NOTE (MONAI 1.6): Invertd reads the transform trace and the metadata
    # from the "image" MetaTensor itself, so meta_keys/orig_meta_keys are
    # left as None instead of pointing at "image_meta_dict" (a dict entry
    # that no longer exists by default in MONAI 1.6).
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                meta_keys=None,
                orig_meta_keys=None,
                meta_key_postfix="meta_dict",
                nearest_interp=True,  # nearest neighbour: keep labels 0-3 exact
                to_tensor=True,
            ),
            # NOTE (MONAI 1.6): the metadata travels ON the MetaTensor itself
            # (pred.meta), not in a separate "pred_meta_dict" dict entry,
            # so we pass meta_keys=None and let SaveImage read pred.meta.
            SaveImaged(
                keys="pred",
                meta_keys=None,
                meta_key_postfix=None,
                output_dir=args.output_dir,
                output_postfix="seg",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,  # save as outputs/<name>_seg.nii.gz
            ),
        ]
    )

    # argmax over the 4 output channels -> one label map (0-3)
    post_pred = AsDiscrete(argmax=True)

    model = build_model(args.checkpoint, device)

    if os.path.isdir(args.input):
        input_files = sorted(glob.glob(os.path.join(args.input, "*.nii.gz")))
    else:
        input_files = [args.input]
    print(f"Found {len(input_files)} volume(s) to segment.")

    with torch.no_grad():  # no gradients needed for inference
        for input_file in input_files:
            name = os.path.splitext(os.path.splitext(os.path.basename(input_file))[0])[0]
            print(f"Segmenting {input_file} ...")
            data = infer_transforms({"image": input_file})
            image = data["image"].unsqueeze(0).to(device)  # add batch dimension
            outputs = sliding_window_inference(
                image, tuple(args.roi_size), args.sw_batch_size, model
            )
            pred = post_pred(outputs[0])  # drop batch dim, take winning class
            # MONAI 1.6 keeps image metadata (affine, spacing, the list of
            # applied transforms) ON the MetaTensor itself, not in a separate
            # "image_meta_dict" dict entry. Invertd finds the transform trace
            # through the "image" MetaTensor, but only if the prediction is
            # a MetaTensor too -- so wrap it before inverting.
            pred = convert_to_tensor(pred, track_meta=True)
            data["pred"] = pred
            post_transforms(data)
            print(f"  saved: {os.path.join(args.output_dir, name + '_seg.nii.gz')}")

    print("Done.")


if __name__ == "__main__":
    main()
