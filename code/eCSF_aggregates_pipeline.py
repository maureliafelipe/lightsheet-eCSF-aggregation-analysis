
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lightsheet_demo_pipeline.py

End-to-end demo pipeline for the light-sheet paper repository.

What this script can do
-----------------------
1. Reconstruct 3D probability stacks from Ilastik per-slice exports.
2. Threshold probability stacks into binary segmentations.
3. Apply 2D erosion to cellmask segmentation (slice-by-slice).
4. Split the eroded cellmask into two compartments using phalloidin:
      - neuroepithelial wall
      - aggregated network
5. Quantify 3D surface, volume, projected area, and surface/volume.
6. Export CSV tables and summary plots.
7. Export example figures so a new user can visually inspect what the
   pipeline is doing on their own images.

How a new user should use this script
-------------------------------------
A. Edit ONLY the CONFIG section below:
   - BASE_DIR
   - voxel sizes
   - thresholds
   - file naming if your names differ
   - example Z slices and display settings

B. Run the script:
       python lightsheet_demo_pipeline.py

C. Outputs will be written into:
       <BASE_DIR>/demo results/
       <BASE_DIR>/demo results/quantification/
       <BASE_DIR>/demo results/examples/

Expected inputs already available in the repository
---------------------------------------------------
1. Raw subset stacks in:
       data sample/
2. Ilastik probability slices already exported as TIFFs in:
       demo results/
   Expected examples:
       E1_cellmask_z0_Probabilities.tif
       E1_phallo_z0_Probabilities.tif
       ...

Notes
-----
- This script assumes the current demo conventions:
    * cellmask final masks / eroded masks / wall / network -> signal BLACK
    * phalloidin final segmentation -> signal WHITE
- If your biological interpretation of wall and network is swapped,
  set SWAP_WALL_AND_NETWORK = True in the CONFIG section.
"""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


# =========================================================
# CONFIGURATION (EDIT THIS SECTION)
# =========================================================
BASE_DIR = Path("/Users/fmaurelia/Library/CloudStorage/OneDrive-UniversitéClermontAuvergne/PHD/Github repository")
DATA_DIR = BASE_DIR / "data sample"
RESULTS_DIR = BASE_DIR / "demo results"
QUANT_DIR = RESULTS_DIR / "quantification"
EXAMPLES_DIR = RESULTS_DIR / "examples"

# Create output folders
for _d in [RESULTS_DIR, QUANT_DIR, EXAMPLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

EMBRYOS = ["E1", "E2", "E3"]

# Voxel sizes from original CZI metadata
PIXEL_SIZES = {
    "E1": {"x_um": 2.609890922263899, "y_um": 2.609890922263899, "z_um": 5.1878006546400135},
    "E2": {"x_um": 2.606761966323059, "y_um": 2.606761966323059, "z_um": 5.957028609336942},
    "E3": {"x_um": 2.606761966323059, "y_um": 2.606761966323059, "z_um": 5.957028609336942},
}

# Thresholds for probability-to-segmentation
CELLMASK_THRESHOLDS = {
    "E1": 0.65,
    "E2": 0.80,
    "E3": 0.80,
}
PHALLO_THRESHOLDS = {
    "E1": 0.80,
    "E2": 0.80,
    "E3": 0.80,
}

# Signal channel in Ilastik probability TIFFs
SIGNAL_CHANNEL = 1

# 2D erosion of cellmask before compartment split / quantification
CELLMASK_EROSION_ITERATIONS = 2

# Downsample factor in XY for surface approximation
DS_XY_SURFACE = 2

# If your visual QC shows wall/network were biologically swapped,
# set this to True and the script will swap the labels.
SWAP_WALL_AND_NETWORK = False

# Example Z slices to export for QC figures
Z_TO_SHOW = {
    "E1": 0,
    "E2": 0,
    "E3": 0,
}

# Display settings for example raw figures
DISPLAY_SETTINGS = {
    "cellmask": {
        "E1": {"vmin": 1, "vmax": 200, "gamma": 0.3},
        "E2": {"vmin": 50, "vmax": 1200, "gamma": 1.2},
        "E3": {"vmin": 0, "vmax": 3000, "gamma": 1.0},
    },
    "phalloidin": {
        "E1": {"vmin": 1, "vmax": 200, "gamma": 0.3},
        "E2": {"vmin": 50, "vmax": 1200, "gamma": 1.2},
        "E3": {"vmin": 0, "vmax": 3000, "gamma": 1.0},
    },
}

# Toggle sections if needed
RUN_RECONSTRUCT_AND_THRESHOLD = True
RUN_CELLMASK_EROSION = True
RUN_SPLIT_WALL_NETWORK = True
RUN_QUANTIFICATION = True
RUN_EXAMPLE_EXPORTS = True


# =========================================================
# HELPERS
# =========================================================
def adjust_image(img: np.ndarray, vmin: float, vmax: float, gamma: float = 1.0) -> np.ndarray:
    img = img.astype(np.float32)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)
    img = np.power(img, gamma)
    return img


def extract_z_index(filename: str) -> int:
    m = re.search(r"_z(\d+)_Probabilities\.tif$", filename)
    if m is None:
        raise ValueError(f"Could not extract z index from: {filename}")
    return int(m.group(1))


def normalise_probability(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        return arr
    if arr.max() <= 255:
        return arr / 255.0
    if arr.max() <= 65535:
        return arr / 65535.0
    return arr / arr.max()


def mask_from_black_signal(img: np.ndarray) -> np.ndarray:
    return img == img.min()


def mask_from_white_signal(img: np.ndarray) -> np.ndarray:
    return img > 0


def mask_to_black_signal_uint8(mask_bool: np.ndarray) -> np.ndarray:
    return np.where(mask_bool, 0, 255).astype(np.uint8)


def mask_to_white_signal_uint8(mask_bool: np.ndarray) -> np.ndarray:
    return np.where(mask_bool, 255, 0).astype(np.uint8)


def erode_mask_2d(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    if iterations is None or iterations == 0:
        return mask
    return binary_erosion(mask, iterations=iterations)


def get_signal_mask(img: np.ndarray, signal_is_white: bool = True) -> np.ndarray:
    if signal_is_white:
        return img > 0
    return img == img.min()


# =========================================================
# STEP 1. RECONSTRUCT PROBABILITY STACKS + THRESHOLD
# =========================================================
def reconstruct_probability_and_segment(channel: str, thresholds: Dict[str, float]) -> Dict[str, Dict[str, Path]]:
    """
    channel: 'cellmask' or 'phalloidin'
    Returns file paths per embryo.
    """
    outputs = {}

    for embryo in EMBRYOS:
        threshold = thresholds[embryo]

        if channel == "cellmask":
            prefix = f"{embryo}_cellmask_z"
            prob_stack_path = RESULTS_DIR / f"{embryo}_cellmask_probability_stack.tif"
            seg_output_path = RESULTS_DIR / f"{embryo}_cellmask_segmentation_thr{str(threshold).replace('.', '')}.tif"
        elif channel == "phalloidin":
            prefix = f"{embryo}_phallo_z"
            prob_stack_path = RESULTS_DIR / f"{embryo}_phalloidin_probability_stack.tif"
            seg_output_path = RESULTS_DIR / f"{embryo}_phalloidin_segmentation_thr{str(threshold).replace('.', '')}.tif"
        else:
            raise ValueError("channel must be 'cellmask' or 'phalloidin'")

        prob_files = [
            RESULTS_DIR / f
            for f in os.listdir(RESULTS_DIR)
            if f.startswith(prefix) and f.endswith("_Probabilities.tif")
        ]

        if len(prob_files) == 0:
            raise FileNotFoundError(
                f"No probability slice files found for {embryo} | {channel} in: {RESULTS_DIR}"
            )

        prob_files = sorted(prob_files, key=lambda p: extract_z_index(p.name))
        print(f"
[{channel}] {embryo}: found {len(prob_files)} slices")

        prob_slices = []
        for i, f in enumerate(prob_files):
            prob = normalise_probability(tiff.imread(f))
            prob_slices.append(prob)
            if i == 0:
                print(f"  Example probability slice shape: {prob.shape}")
                print(f"  Example probability min/max: {float(prob.min())} / {float(prob.max())}")

        prob_stack = np.stack(prob_slices, axis=0)  # (Z, Y, X, C)
        if prob_stack.ndim != 4:
            raise ValueError(f"Expected 4D prob stack, got {prob_stack.shape}")

        signal_prob = prob_stack[:, :, :, SIGNAL_CHANNEL]
        seg_stack = (signal_prob > threshold).astype(np.uint8) * 255

        tiff.imwrite(prob_stack_path, prob_stack.astype(np.float32), metadata={"axes": "ZYXC"})
        tiff.imwrite(seg_output_path, seg_stack, metadata={"axes": "ZYX"})

        print(f"  Saved probability stack: {prob_stack_path}")
        print(f"  Saved segmentation: {seg_output_path}")

        outputs[embryo] = {
            "prob_stack": prob_stack_path,
            "segmentation": seg_output_path,
        }

    return outputs


# =========================================================
# STEP 2. 2D EROSION OF CELLMASK (SIGNAL BLACK OUTPUT)
# =========================================================
def erode_cellmask_outputs(cellmask_seg_paths: Dict[str, Path]) -> Dict[str, Path]:
    out_paths = {}

    for embryo in EMBRYOS:
        seg_path = cellmask_seg_paths[embryo]
        threshold_str = "065" if embryo == "E1" else "08"
        out_path = RESULTS_DIR / f"{embryo}_cellmask_segmentation_thr{threshold_str}_eroded{CELLMASK_EROSION_ITERATIONS}_2D_blackSignal.tif"

        seg_stack = tiff.imread(seg_path)
        seg_fg = mask_from_white_signal(seg_stack)  # thresholded cellmask is white signal

        Z, Y, X = seg_fg.shape
        eroded_fg = np.zeros_like(seg_fg, dtype=bool)
        for z in range(Z):
            eroded_fg[z] = erode_mask_2d(seg_fg[z], iterations=CELLMASK_EROSION_ITERATIONS)

        eroded_uint8 = mask_to_black_signal_uint8(eroded_fg)
        tiff.imwrite(out_path, eroded_uint8, metadata={"axes": "ZYX"})

        print(f"
[cellmask erosion] {embryo}")
        print(f"  Input:  {seg_path}")
        print(f"  Output: {out_path}")
        print(f"  Fraction before: {float(seg_fg.mean())}")
        print(f"  Fraction after:  {float(eroded_fg.mean())}")

        out_paths[embryo] = out_path

    return out_paths


# =========================================================
# STEP 3. SPLIT INTO WALL / NETWORK
# =========================================================
def split_wall_network(cellmask_eroded_paths: Dict[str, Path], phallo_seg_paths: Dict[str, Path]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    wall_paths = {}
    network_paths = {}

    for embryo in EMBRYOS:
        cellmask_img = tiff.imread(cellmask_eroded_paths[embryo])
        phallo_img = tiff.imread(phallo_seg_paths[embryo])

        if cellmask_img.shape != phallo_img.shape:
            raise ValueError(
                f"Shape mismatch for {embryo}: {cellmask_img.shape} vs {phallo_img.shape}"
            )

        cellmask_fg = mask_from_black_signal(cellmask_img)  # signal black
        phallo_fg = mask_from_white_signal(phallo_img)      # signal white

        # Default interpretation:
        #   wall     = overlap
        #   network  = difference
        wall_fg = cellmask_fg & phallo_fg
        network_fg = cellmask_fg & (~phallo_fg)

        if SWAP_WALL_AND_NETWORK:
            wall_fg, network_fg = network_fg, wall_fg

        wall_path = RESULTS_DIR / f"{embryo}_neuroepithelial_wall_blackSignal.tif"
        network_path = RESULTS_DIR / f"{embryo}_aggregated_network_blackSignal.tif"

        tiff.imwrite(wall_path, mask_to_black_signal_uint8(wall_fg), metadata={"axes": "ZYX"})
        tiff.imwrite(network_path, mask_to_black_signal_uint8(network_fg), metadata={"axes": "ZYX"})

        print(f"
[split] {embryo}")
        print(f"  Wall fraction:    {float(wall_fg.mean())}")
        print(f"  Network fraction: {float(network_fg.mean())}")
        print(f"  Saved wall:    {wall_path}")
        print(f"  Saved network: {network_path}")

        wall_paths[embryo] = wall_path
        network_paths[embryo] = network_path

    return wall_paths, network_paths


# =========================================================
# STEP 4. QUANTIFICATION
# =========================================================
class StackReader:
    def __init__(self, path: Path):
        self.path = str(path)
        self.arr = np.squeeze(tiff.imread(self.path))

        print("
Opening:")
        print(self.path)
        print("Raw loaded shape:", self.arr.shape)
        print("dtype:", self.arr.dtype)

        if self.arr.ndim == 2:
            self.arr = self.arr[np.newaxis, :, :]
        elif self.arr.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported shape after squeeze: {self.arr.shape}")

        self.Z, self.Y, self.X = self.arr.shape
        print("Detected shape:", (self.Z, self.Y, self.X))
        print("Mode: in-memory 3D")

    def read_z(self, z: int) -> np.ndarray:
        return self.arr[z]

    def close(self):
        pass


def compute_metrics_streaming(
    path: Path,
    Z_use: int,
    pixel_x_um: float,
    pixel_y_um: float,
    pixel_z_um: float,
    signal_is_white: bool = True,
    ds_xy_surface: int = 2,
):
    reader = StackReader(path)

    if Z_use > reader.Z:
        raise ValueError(f"Z_use={Z_use} larger than file Z={reader.Z}")

    voxel_volume_um3 = pixel_z_um * pixel_y_um * pixel_x_um
    pixel_area_um2 = pixel_y_um * pixel_x_um

    volume_voxels = 0
    projection = np.zeros((reader.Y, reader.X), dtype=bool)

    dz = pixel_z_um
    dy = pixel_y_um * ds_xy_surface
    dx = pixel_x_um * ds_xy_surface

    face_area_z = dy * dx
    face_area_y = dz * dx
    face_area_x = dz * dy

    surface_um2 = 0.0
    prev_ds = None

    for z in range(Z_use):
        img = reader.read_z(z)
        if img.ndim != 2:
            raise ValueError(f"read_z({z}) from {path} returned shape {img.shape}, expected 2D")

        mask_full = get_signal_mask(img, signal_is_white=signal_is_white)

        volume_voxels += int(mask_full.sum())
        projection |= mask_full

        mask_ds = mask_full[::ds_xy_surface, ::ds_xy_surface]

        # Z-direction faces
        if prev_ds is None:
            surface_um2 += mask_ds.sum() * face_area_z
        else:
            surface_um2 += np.logical_xor(mask_ds, prev_ds).sum() * face_area_z

        # Y-direction faces
        surface_um2 += mask_ds[0, :].sum() * face_area_y
        surface_um2 += mask_ds[-1, :].sum() * face_area_y
        surface_um2 += np.logical_xor(mask_ds[1:, :], mask_ds[:-1, :]).sum() * face_area_y

        # X-direction faces
        surface_um2 += mask_ds[:, 0].sum() * face_area_x
        surface_um2 += mask_ds[:, -1].sum() * face_area_x
        surface_um2 += np.logical_xor(mask_ds[:, 1:], mask_ds[:, :-1]).sum() * face_area_x

        prev_ds = mask_ds

    if prev_ds is not None:
        surface_um2 += prev_ds.sum() * face_area_z

    volume_um3 = volume_voxels * voxel_volume_um3
    projected_area_um2 = projection.sum() * pixel_area_um2

    reader.close()

    return {
        "surface_um2": surface_um2,
        "volume_um3": volume_um3,
        "projected_area_um2": projected_area_um2,
        "surface_to_volume_um_minus1": surface_um2 / volume_um3 if volume_um3 > 0 else np.nan,
    }


def run_quantification(cellmask_eroded_paths: Dict[str, Path], phallo_seg_paths: Dict[str, Path], wall_paths: Dict[str, Path], network_paths: Dict[str, Path]) -> Tuple[pd.DataFrame, Path, Path]:
    paths = {
        embryo: {
            "cellmask_eroded": cellmask_eroded_paths[embryo],
            "phallo": phallo_seg_paths[embryo],
            "wall": wall_paths[embryo],
            "network": network_paths[embryo],
        }
        for embryo in EMBRYOS
    }

    signal_is_white_map = {
        "cellmask_eroded": False,
        "wall": False,
        "network": False,
        "phallo": True,
    }

    rows = []

    for embryo, p in paths.items():
        print("
===================================================")
        print("Processing embryo:", embryo)
        print("===================================================")

        px = PIXEL_SIZES[embryo]["x_um"]
        py = PIXEL_SIZES[embryo]["y_um"]
        pz = PIXEL_SIZES[embryo]["z_um"]

        readers = {k: StackReader(v) for k, v in p.items()}
        XY_shapes = {(r.Y, r.X) for r in readers.values()}
        if len(XY_shapes) != 1:
            raise ValueError(f"XY mismatch for {embryo}: {XY_shapes}")

        Z_use = min(r.Z for r in readers.values())
        for key, r in readers.items():
            if r.Z != Z_use:
                print(f"WARNING: {embryo} {key} has Z={r.Z}, using Z_use={Z_use}")
            r.close()

        metrics = {}
        for compartment in ["cellmask_eroded", "phallo", "wall", "network"]:
            print(f"
Computing metrics for {embryo} | {compartment} ...")
            metrics[compartment] = compute_metrics_streaming(
                p[compartment],
                Z_use=Z_use,
                pixel_x_um=px,
                pixel_y_um=py,
                pixel_z_um=pz,
                signal_is_white=signal_is_white_map[compartment],
                ds_xy_surface=DS_XY_SURFACE,
            )
            gc.collect()

        wall_over_cellmask_volume = (
            metrics["wall"]["volume_um3"] / metrics["cellmask_eroded"]["volume_um3"]
            if metrics["cellmask_eroded"]["volume_um3"] > 0 else np.nan
        )
        network_over_cellmask_volume = (
            metrics["network"]["volume_um3"] / metrics["cellmask_eroded"]["volume_um3"]
            if metrics["cellmask_eroded"]["volume_um3"] > 0 else np.nan
        )
        wall_over_network_volume = (
            metrics["wall"]["volume_um3"] / metrics["network"]["volume_um3"]
            if metrics["network"]["volume_um3"] > 0 else np.nan
        )
        wall_over_cellmask_surface = (
            metrics["wall"]["surface_um2"] / metrics["cellmask_eroded"]["surface_um2"]
            if metrics["cellmask_eroded"]["surface_um2"] > 0 else np.nan
        )
        network_over_cellmask_surface = (
            metrics["network"]["surface_um2"] / metrics["cellmask_eroded"]["surface_um2"]
            if metrics["cellmask_eroded"]["surface_um2"] > 0 else np.nan
        )
        wall_over_network_surface = (
            metrics["wall"]["surface_um2"] / metrics["network"]["surface_um2"]
            if metrics["network"]["surface_um2"] > 0 else np.nan
        )

        rows.append({
            "embryo": embryo,
            "pixel_x_um": px,
            "pixel_y_um": py,
            "pixel_z_um": pz,
            "Z_used": Z_use,
            "surface_ds_xy": DS_XY_SURFACE,
            # cellmask eroded
            "cellmask_eroded_surface_um2": metrics["cellmask_eroded"]["surface_um2"],
            "cellmask_eroded_volume_um3": metrics["cellmask_eroded"]["volume_um3"],
            "cellmask_eroded_projected_area_um2": metrics["cellmask_eroded"]["projected_area_um2"],
            "cellmask_eroded_surface_to_volume_um_minus1": metrics["cellmask_eroded"]["surface_to_volume_um_minus1"],
            # phallo
            "phallo_surface_um2": metrics["phallo"]["surface_um2"],
            "phallo_volume_um3": metrics["phallo"]["volume_um3"],
            "phallo_projected_area_um2": metrics["phallo"]["projected_area_um2"],
            "phallo_surface_to_volume_um_minus1": metrics["phallo"]["surface_to_volume_um_minus1"],
            # wall
            "wall_surface_um2": metrics["wall"]["surface_um2"],
            "wall_volume_um3": metrics["wall"]["volume_um3"],
            "wall_projected_area_um2": metrics["wall"]["projected_area_um2"],
            "wall_surface_to_volume_um_minus1": metrics["wall"]["surface_to_volume_um_minus1"],
            # network
            "network_surface_um2": metrics["network"]["surface_um2"],
            "network_volume_um3": metrics["network"]["volume_um3"],
            "network_projected_area_um2": metrics["network"]["projected_area_um2"],
            "network_surface_to_volume_um_minus1": metrics["network"]["surface_to_volume_um_minus1"],
            # derived fractions
            "wall_over_cellmask_volume": wall_over_cellmask_volume,
            "network_over_cellmask_volume": network_over_cellmask_volume,
            "wall_over_network_volume": wall_over_network_volume,
            "wall_over_cellmask_surface": wall_over_cellmask_surface,
            "network_over_cellmask_surface": network_over_cellmask_surface,
            "wall_over_network_surface": wall_over_network_surface,
        })

        print(f"
Done: {embryo}")
        print("Wall / cellmask volume:", wall_over_cellmask_volume)
        print("Network / cellmask volume:", network_over_cellmask_volume)
        print("Wall / network volume:", wall_over_network_volume)

    df = pd.DataFrame(rows)
    df.to_csv(csv_output, index=False)
    print("
Saved CSV:")
    print(csv_output)

    # -------------------------------
    # PLOTS
    # -------------------------------
    df_plot = df.copy()
    df_plot["embryo"] = pd.Categorical(df_plot["embryo"], categories=EMBRYOS, ordered=True)
    df_plot = df_plot.sort_values("embryo").reset_index(drop=True)

    embryos = df_plot["embryo"].astype(str).values
    x = np.arange(len(embryos))
    width = 0.25

    surface_scale = 1e6
    volume_scale = 1e9

    # Figure 1: absolute metrics
    fig_abs, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    wall_surface = df_plot["wall_surface_um2"].values / surface_scale
    network_surface = df_plot["network_surface_um2"].values / surface_scale
    ax.bar(x - width/2, wall_surface, width, label="Neuroepithelial wall", color="royalblue", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, network_surface, width, label="Aggregated network", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("3D surface area (×10⁶ µm²)")
    ax.set_xlabel("Embryo")
    ax.set_title("3D surface area")
    ax.legend(frameon=False)

    ax = axes[1]
    wall_volume = df_plot["wall_volume_um3"].values / volume_scale
    network_volume = df_plot["network_volume_um3"].values / volume_scale
    ax.bar(x - width/2, wall_volume, width, label="Neuroepithelial wall", color="royalblue", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, network_volume, width, label="Aggregated network", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("Volume (×10⁹ µm³)")
    ax.set_xlabel("Embryo")
    ax.set_title("3D volume")
    ax.legend(frameon=False)

    ax = axes[2]
    wall_sv = df_plot["wall_surface_to_volume_um_minus1"].values
    network_sv = df_plot["network_surface_to_volume_um_minus1"].values
    ax.bar(x - width/2, wall_sv, width, label="Neuroepithelial wall", color="royalblue", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, network_sv, width, label="Aggregated network", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("Surface-to-volume ratio (µm⁻¹)")
    ax.set_xlabel("Embryo")
    ax.set_title("Surface / volume")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig_abs.savefig(absolute_metrics_path, dpi=600, bbox_inches="tight", transparent=False)
    plt.show()

    # Figure 2: fractions / ratios
    fig_frac, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.bar(x - width/2, df_plot["wall_over_cellmask_volume"].values, width, label="Wall / cellmask", color="royalblue", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, df_plot["network_over_cellmask_volume"].values, width, label="Network / cellmask", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("Volume fraction")
    ax.set_xlabel("Embryo")
    ax.set_title("Volume fraction of eroded cellmask")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x - width/2, df_plot["wall_over_cellmask_surface"].values, width, label="Wall / cellmask", color="royalblue", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, df_plot["network_over_cellmask_surface"].values, width, label="Network / cellmask", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("Surface fraction")
    ax.set_xlabel("Embryo")
    ax.set_title("Surface fraction of eroded cellmask")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.bar(x, df_plot["wall_over_network_volume"].values, width=0.45, color="purple", edgecolor="black", alpha=0.8)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(embryos)
    ax.set_ylabel("Wall / network volume ratio")
    ax.set_xlabel("Embryo")
    ax.set_title("Relative compartment size")

    plt.tight_layout()
    fig_frac.savefig(fraction_metrics_path, dpi=600, bbox_inches="tight", transparent=False)
    plt.show()

    print("Saved absolute metrics figure:")
    print(absolute_metrics_path)
    print("Saved fraction metrics figure:")
    print(fraction_metrics_path)

    return df, absolute_metrics_path, fraction_metrics_path


from pathlib import Path
from typing import Dict

# =========================================================
# STEP 5. EXAMPLE EXPORTS FOR NEW USERS
# =========================================================
def get_segmentation_paths(embryo: str) -> Dict[str, Path]:
    cellmask_thr_str = "065" if embryo == "E1" else "08"
    phallo_thr_str = "08"

    return {
        "raw_cellmask": DATA_DIR / f"{embryo}_cellmask_subset.tif",
        "raw_phalloidin": DATA_DIR / f"{embryo}_phalloidin_subset.tif",
        "cellmask_prob": RESULTS_DIR / f"{embryo}_cellmask_probability_stack.tif",
        "phallo_prob": RESULTS_DIR / f"{embryo}_phalloidin_probability_stack.tif",
        "cellmask_seg": RESULTS_DIR / f"{embryo}_cellmask_segmentation_thr{cellmask_thr_str}.tif",
        "phallo_seg": RESULTS_DIR / f"{embryo}_phalloidin_segmentation_thr{phallo_thr_str}.tif",
        "cellmask_eroded": RESULTS_DIR / f"{embryo}_cellmask_segmentation_thr{cellmask_thr_str}_eroded{CELLMASK_EROSION_ITERATIONS}_2D_blackSignal.tif",
        "wall": RESULTS_DIR / f"{embryo}_neuroepithelial_wall_blackSignal.tif",
        "network": RESULTS_DIR / f"{embryo}_aggregated_network_blackSignal.tif",
    }


def export_example_figures():
    # -----------------------------------------------------
    # 1) cellmask + phalloidin raw / probability / segmentation
    # -----------------------------------------------------
    for channel in ["cellmask", "phalloidin"]:
        fig, axes = plt.subplots(len(EMBRYOS), 4, figsize=(16, 4 * len(EMBRYOS)))

        if len(EMBRYOS) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, embryo in enumerate(EMBRYOS):
            paths = get_segmentation_paths(embryo)
            z = Z_TO_SHOW[embryo]

            if channel == "cellmask":
                raw_stack = tiff.imread(paths["raw_cellmask"])
                prob_stack = tiff.imread(paths["cellmask_prob"])
                seg_stack = tiff.imread(paths["cellmask_seg"])
            else:
                raw_stack = tiff.imread(paths["raw_phalloidin"])
                prob_stack = tiff.imread(paths["phallo_prob"])
                seg_stack = tiff.imread(paths["phallo_seg"])

            raw = raw_stack[z]
            prob_fg = prob_stack[z, :, :, SIGNAL_CHANNEL]
            seg = seg_stack[z]

            params = DISPLAY_SETTINGS[channel][embryo]
            raw_adj = adjust_image(
                raw,
                params["vmin"],
                params["vmax"],
                params["gamma"]
            )

            # raw
            axes[i, 0].imshow(raw_adj, cmap="gray")
            axes[i, 0].set_title(f"{embryo} | raw | z={z}")
            axes[i, 0].axis("off")

            # probability
            axes[i, 1].imshow(prob_fg, cmap="magma", vmin=0, vmax=1)
            axes[i, 1].set_title(f"{embryo} | probability | z={z}")
            axes[i, 1].axis("off")

            # segmentation
            axes[i, 2].imshow(seg, cmap="gray")
            axes[i, 2].set_title(f"{embryo} | segmentation | z={z}")
            axes[i, 2].axis("off")

            # overlay
            axes[i, 3].imshow(raw_adj, cmap="gray")
            axes[i, 3].imshow(seg > 0, cmap="spring", alpha=0.35)
            axes[i, 3].set_title(f"{embryo} | overlay | z={z}")
            axes[i, 3].axis("off")

        plt.tight_layout()
        out = EXAMPLES_DIR / f"example_{channel}_raw_prob_seg_overlay.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved example figure: {out}")

    # -----------------------------------------------------
    # 2) wall vs network compartment figure
    # -----------------------------------------------------
    fig, axes = plt.subplots(len(EMBRYOS), 4, figsize=(16, 4 * len(EMBRYOS)))

    if len(EMBRYOS) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, embryo in enumerate(EMBRYOS):
        paths = get_segmentation_paths(embryo)
        z = Z_TO_SHOW[embryo]

        cellmask_img = tiff.imread(paths["cellmask_eroded"])[z]
        phallo_img = tiff.imread(paths["phallo_seg"])[z]
        wall_img = tiff.imread(paths["wall"])[z]
        network_img = tiff.imread(paths["network"])[z]

        axes[i, 0].imshow(cellmask_img, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"{embryo} | cellmask eroded | z={z}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(phallo_img, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(f"{embryo} | phallo segmented | z={z}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(wall_img, cmap="gray", vmin=0, vmax=255)
        axes[i, 2].set_title(f"{embryo} | neuroepithelial wall | z={z}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(network_img, cmap="gray", vmin=0, vmax=255)
        axes[i, 3].set_title(f"{embryo} | aggregated network | z={z}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out = EXAMPLES_DIR / "example_wall_vs_aggregated_network.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved example figure: {out}")


# =========================================================
# MAIN
# =========================================================
def main():
    # Keep track of outputs so the script can be run from scratch.
    cellmask_outputs = {}
    phallo_outputs = {}

    if RUN_RECONSTRUCT_AND_THRESHOLD:
        cellmask_outputs = reconstruct_probability_and_segment(
            "cellmask",
            CELLMASK_THRESHOLDS
        )
        phallo_outputs = reconstruct_probability_and_segment(
            "phalloidin",
            PHALLO_THRESHOLDS
        )
    else:
        # Rebuild path maps assuming outputs already exist
        for embryo in EMBRYOS:
            cellmask_thr_str = "065" if embryo == "E1" else "08"
            phallo_thr_str = "08"

            cellmask_outputs[embryo] = {
                "prob_stack": RESULTS_DIR / f"{embryo}_cellmask_probability_stack.tif",
                "segmentation": RESULTS_DIR / f"{embryo}_cellmask_segmentation_thr{cellmask_thr_str}.tif",
            }

            phallo_outputs[embryo] = {
                "prob_stack": RESULTS_DIR / f"{embryo}_phalloidin_probability_stack.tif",
                "segmentation": RESULTS_DIR / f"{embryo}_phalloidin_segmentation_thr{phallo_thr_str}.tif",
            }

    if RUN_CELLMASK_EROSION:
        cellmask_eroded_paths = erode_cellmask_outputs(
            {e: cellmask_outputs[e]["segmentation"] for e in EMBRYOS}
        )
    else:
        cellmask_eroded_paths = {
            "E1": RESULTS_DIR / f"E1_cellmask_segmentation_thr065_eroded{CELLMASK_EROSION_ITERATIONS}_2D_blackSignal.tif",
            "E2": RESULTS_DIR / f"E2_cellmask_segmentation_thr08_eroded{CELLMASK_EROSION_ITERATIONS}_2D_blackSignal.tif",
            "E3": RESULTS_DIR / f"E3_cellmask_segmentation_thr08_eroded{CELLMASK_EROSION_ITERATIONS}_2D_blackSignal.tif",
        }

    if RUN_SPLIT_WALL_NETWORK:
        wall_paths, network_paths = split_wall_network(
            cellmask_eroded_paths,
            {e: phallo_outputs[e]["segmentation"] for e in EMBRYOS},
        )
    else:
        wall_paths = {
            e: RESULTS_DIR / f"{e}_neuroepithelial_wall_blackSignal.tif"
            for e in EMBRYOS
        }
        network_paths = {
            e: RESULTS_DIR / f"{e}_aggregated_network_blackSignal.tif"
            for e in EMBRYOS
        }

    if RUN_QUANTIFICATION:
        run_quantification(
            cellmask_eroded_paths,
            {e: phallo_outputs[e]["segmentation"] for e in EMBRYOS},
            wall_paths,
            network_paths,
        )

    if RUN_EXAMPLE_EXPORTS:
        export_example_figures()

    print("\\nPipeline completed successfully.")
    print(f"Check outputs in: {RESULTS_DIR}")
    print(f"Quantification in: {QUANT_DIR}")
    print(f"Examples in: {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()