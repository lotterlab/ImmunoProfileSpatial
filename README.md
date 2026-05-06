# ImmunoProfileSpatial

This repository accompanies the paper **"Graph neural network modeling of spatial 
tumor-immune interactions identifies prognostic cellular niches in non-small cell 
lung cancer"** (*npj Precision Oncology*, 2026. https://doi.org/10.1038/s41698-026-01314-3).
It provides code for constructing spatial graphs from multiplex immunofluorescence (mIF) data acquired as part of the ImmunoProfile project at DFCI and training a graph neural network (GNN) to predict patient survival based on localized tumor–immune interactions (based on the [SPACE-GM framework](https://gitlab.com/enable-medicine-public/space-gm/-/tree/main) by Wu et al.

Data link: https://doi.org/10.7303/syn52596661

---

## Setup
The `space-gm` directory is a submodule that contains the GNN implementation used in our experiments. Clone or install it separately if it is not present.

Create the conda environment listed in `immunoprofilespatialenv.yml`:


```bash
conda env create -f immunoprofilespatialenv.yml
```
---

## Workflow

**Preprocess ROIs:** Use scripts/preprocessing.py and scripts/roi_qc.py to convert raw single-cell CSV files into graphs (Voronoi polygons, Delaunay triangulation) and to perform ROI quality control.

**Create dataset splits:** scripts/create_dataset_splits.py generates training/validation/test splits at the case or ROI level. Example splits are provided in data/experiment_split.json.

**Generate graph/subgraph datasets:** Run scripts/generate_subgraph_dataset.py to build full graphs and spatial neighborhood subgraphs for each split. Paths and dataset parameters are controlled via JSON files in configs/.

**Train the model:** scripts/train_model.py trains a SPACE‑GM model on the generated subgraphs. Training parameters (batch size, learning rate, etc.) are defined in configs/train_params.json.

**Evaluation:** Model outputs include per-neighborhood survival predictions that can be aggregated at the patient level. Example evaluation utilities are provided in scripts/utils_spatial.py.

**Analysis:** Subgraph manipulations described in the paper are implemented as transformations and can be found in additional_transforms.py. Visualization utilities can be found in utils_spatial.py.

Note: parts of the preprocessing pipeline and the training scripts may still be incomplete. We will update the repository as additional code becomes available.

---
## Data

The dataset used in this paper is derived from the publicly available 
[ImmunoProfile project](https://doi.org/10.7303/syn52596661) on Synapse 
(syn52596661). It is **not** included in this repository, but can be freely 
downloaded from Synapse after account registration.

### Downloading the data

This paper uses the **NSCLC subset** of the pan-cancer ImmunoProfile dataset. 
Two files are needed:

**1. Metadata file**  
Available as a [Synapse table](https://www.synapse.org/Synapse:syn69058119/tables/).  
Filter to NSCLC patients: `oncotree_metamain == 'Non-Small Cell Lung Cancer'`.

**2. Single-cell parquet file**  
Available as a parquet file. Download [single_cells.parquet](https://www.synapse.org/Synapse:syn69057790.draft/datasets/) from Synapse.  
This file contains ~39 million spatially-resolved cells across all cancer types. 
Filter to the `case_id` values from the NSCLC metadata, keeping only 
`region_label == 'InnerTumor'` cells.

### Data format expected by the pipeline

After filtering, the single-cell data should have the following columns, 
which map directly to the graph construction inputs:

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | int64 | De-identified patient ID |
| `roi_id` | object | ROI identifier |
| `region_label` | object | segmented region identifier |
| `cell_x` | int64 | Cell centroid x |
| `cell_y` | int64 | Cell centroid y |
| `cd8` | bool | CD8 marker positivity |
| `pd1` | bool | PD-1 marker positivity |
| `pdl1` | bool | PD-L1 marker positivity |
| `foxp3` | bool | FOXP3 marker positivity |
| `tumor` | bool | Cytokeratin (tumor) marker positivity |

The `data/` directory in this repository contains minimal placeholder files 
illustrating the expected formats:
- `full_graph_labels.csv` — survival labels per ROI
- `markers.json` — cell type and feature definitions
- `experiment_split.json` — the exact train/validation/test split used in the paper 
(305/72/129 patients, stratified by survival status)


