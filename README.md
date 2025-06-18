# ImmunoProfileSpatial

This repository accompanies the paper  **“Graph neural network modeling of spatial tumor‑immune interactions identifies prognostic cellular niches in non‑small cell lung cancer.”**  
It provides code for constructing spatial graphs from multiplex immunofluorescence (mIF) data acquired as part of the ImmunoProfile project at DFCI and training a graph neural network (GNN) to predict patient survival based on localized tumor–immune interactions (based on the [SPACE-GM framework](https://gitlab.com/enable-medicine-public/space-gm/-/tree/main) by Wu et al.



---

## Setup
The `space-gm` directory is a submodule that contains the GNN implementation used in our experiments. Clone or install it separately if it is not present.

Create the conda environment listed in `immunoprofilespatialenv.yml`:

```bash
conda env create -f immunoprofilespatialenv.yml```


---

## Workflow

**Preprocess ROIs:** Use scripts/preprocessing.py and scripts/roi_qc.py to convert raw single-cell CSV files into graphs (Voronoi polygons, Delaunay triangulation) and to perform ROI quality control.
**Create dataset splits:** scripts/create_dataset_splits.py generates training/validation/test splits at the case or ROI level. Example splits are provided in data/experiment_split.json.
**Generate graph/subgraph datasets:** Run scripts/generate_subgraph_dataset.py to build full graphs and spatial neighborhood subgraphs for each split. Paths and dataset parameters are controlled via JSON files in configs/.
**Train the model:** scripts/train_model.py trains a SPACE‑GM model on the generated subgraphs. Training parameters (batch size, learning rate, etc.) are defined in configs/train_params.json.
**Evaluation:** Model outputs include per-neighborhood survival predictions that can be aggregated at the patient level. Example evaluation utilities are provided in scripts/utils_spatial.py.
**Analysis:** Subgraph manipulations described in the paper are implemented as transformations and can be found in TBD
Note: parts of the preprocessing pipeline and the training scripts may still be incomplete. We will update the repository as additional code becomes available.

---
## Data

The dataset used in the paper is derived from the ImmunoProfile study (Dana‑Farber Cancer Institute) and is not included in this repository. The data/ directory provides minimal placeholder files to illustrate expected formats:

full_graph_labels.csv – survival labels per ROI
markers.json – list of cell types and features
experiment_split.json – example train/validation/test split
Users must supply their own ROI-level single-cell data in the required format to reproduce the analyses.

