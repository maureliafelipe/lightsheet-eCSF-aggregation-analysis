# Light-sheet segmentation and compartment quantification demo

This repository contains a reproducible demo of the image analysis workflow used for the light-sheet paper.  
It shows how 3D subset stacks from multiple embryos were processed from raw images to probability maps, binary segmentations, and quantitative compartment-level measurements.

## Associated publication

This code was used for the analysis presented in the following publication:

**Structural and Molecular Characterization of the Chick Embryonic Cerebrospinal Fluid reveals a luminal extracellular matrix network containing SCO-spondin**

**Felipe Maurelia**<sup>1,4</sup>\*, **Jaime Aguayo**<sup>1</sup>\*, **Francesca Thiele**<sup>1</sup>, **Maryori González**<sup>1</sup>, **Benjamín Molina-Chavez**<sup>1</sup>, **Vania Sepúlveda**<sup>1</sup>, **Antonia Recabal**<sup>1</sup>, **Marcela Torrejón**<sup>2</sup>, **Carlos Farkas**<sup>3</sup>, **Charlene Guillot**<sup>4</sup>, **Ángel Gato**<sup>5</sup>, **Teresa Caprile**<sup>1</sup>

This repository contains the code used to study a molecular complex in the embryonic chick cerebrospinal fluid (eCSF) at developmental stage HH25, following **CellMask-488 injection** and **phalloidin-633 fluorolabelling**.

## Data availability

The full imaging data associated with this repository are archived in Zenodo:
[Zenodo DOI 10.5281/zenodo.21420200](https://doi.org/10.5281/zenodo.21420200)

This Zenodo record contains:

- the full `.czi` files for each embryo
- subsets of selected `z`-slices for some embryos
- the source files corresponding to the examples included in `data_sample/` and used in the demo workflow

The `data_sample/` directory in this repository contains a reduced set of example files for demonstration and testing. The complete per-embryo image files and selected `z`-subsets are available in the Zenodo archive linked above.

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
