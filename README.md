# Light-sheet segmentation and compartment quantification demo

This repository contains a reproducible demo of the image analysis workflow used for the light-sheet paper.  
It shows how 3D subset stacks from multiple embryos were processed from raw images to probability maps, binary segmentations, and quantitative compartment-level measurements.

## Overview

The demo illustrates the following analysis steps:

1. **Input data preparation**
   - 3D subset stacks were selected for three embryos (`E1`, `E2`, `E3`)
   - two channels were analysed:
     - **cellmask**
     - **phalloidin**

2. **Pixel classification with Ilastik**
   - trained Ilastik projects were applied to each Z slice
   - probability maps were exported for both channels

3. **Segmentation from probability maps**
   - probability stacks were reconstructed from the per-slice outputs
   - embryo-specific thresholds were applied to generate binary masks:
     - `E1 cellmask`: threshold = `0.65`
     - `E2 cellmask`: threshold = `0.80`
     - `E3 cellmask`: threshold = `0.80`
     - `phalloidin`: threshold = `0.80`

4. **Cellmask refinement**
   - the cellmask segmentation was refined by applying **2D erosion slice-by-slice**
   - the eroded mask was used for downstream compartment analysis

5. **Compartment definition**
   - the eroded cellmask was combined with the phalloidin segmentation to define:
     - **Neuroepithelial wall**
     - **Aggregated network**

6. **Quantification**
   - 3D surface area
   - 3D volume
   - projected area
   - surface-to-volume ratio
   - relative fractions between compartments

---

## Repository structure

```text
.
├── code/
├── data sample/
├── demo notebook/
├── demo results/
├── Ilastik/
├── results/
├── environment.yml
└── README.md
