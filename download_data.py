#!/usr/bin/env python3
"""
Download the MSD Task01 Brain Tumour dataset (~5.7 GB).

The download comes from MONAI's public mirror of the Medical Segmentation
Decathlon data. No account or registration is needed. The data is released
under the CC BY-SA 4.0 licence.

This dataset is derived from the BraTS challenge: the same four MRI
sequences (FLAIR, T1, T1ce, T2) and the same expert tumour labels as
BraTS, packaged as one 4-channel volume per case. The full BraTS 2021
training set (1,251 cases) is also public, but it needs a Synapse account
and a data-use agreement, and it is far larger -- so this project uses
the MSD Task01 subset (484 training cases), which downloads with one
command and no registration. See README.md for details.

Layout after the download (inside the data folder)::

    data/Task01_BrainTumour/
        imagesTr/     484 training MRI volumes (BRATS_001.nii.gz, ...)
                      each volume has 4 channels, in this order:
                      FLAIR, T1, T1ce, T2
        labelsTr/     484 expert tumour masks (same file names)
                      labels: 0 background, 1 peritumoural oedema,
                      2 necrotic / non-enhancing tumour core,
                      3 enhancing tumour.
                      (These are the MSD labels: the Decathlon remapped
                      the original BraTS labels 1/2/4 to 2/1/3.)
        imagesTs/     266 test MRI volumes (no labels - for the challenge)
        dataset.json  task description and metadata

Usage
-----
    python download_data.py                # downloads into data/
    python download_data.py --out my_data  # downloads into my_data/
"""

import argparse
import os

from monai.apps import download_and_extract

# MONAI's public mirror of the MSD Brain Tumour archive, with its MD5
# checksum (the checksum proves the download was not corrupted).
# Source: monai.apps.datasets.DecathlonDataset (verified 2026-11-15).
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_MD5 = "240a19d752f0d9e9101544901065d872"


def main():
    parser = argparse.ArgumentParser(description="Download MSD Task01 Brain Tumour data.")
    parser.add_argument(
        "--out", default="data",
        help="Folder to download into (default: data).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "Task01_BrainTumour")
    if os.path.isdir(target):
        print(f"Already downloaded: {target}")
        print("Delete that folder first if you want to download it again.")
        return

    print("Downloading Task01_BrainTumour.tar (~5.7 GB). This takes a while ...")
    download_and_extract(
        MSD_URL,
        os.path.join(args.out, "Task01_BrainTumour.tar"),
        args.out,
        MSD_MD5, "md5",
    )
    print(f"Done. Data is in: {target}")
    print("Next step: python train.py --quick   (smoke test)")


if __name__ == "__main__":
    main()
