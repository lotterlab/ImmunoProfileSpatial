# ImmunoProfileSpatial

This repository accompanies the paper  **“Graph neural network modeling of spatial tumor‑immune interactions identifies prognostic cellular niches in non‑small cell lung cancer.”**  
It provides code for constructing spatial graphs from multiplex immunofluorescence (mIF) data acquired as part of the ImmunoProfile project at DFCI and training a graph neural network (GNN) to predict patient survival based on localized tumor–immune interactions (based on the [SPACE-GM framework](https://gitlab.com/enable-medicine-public/space-gm/-/tree/main) by Wu et al.



---

## Setup
The `space-gm` directory is a submodule that contains the GNN implementation used in our experiments. Clone or install it separately if it is not present.

Create the conda environment listed in `immunoprofilespatialenv.yml`:

```bash
conda env create -f immunoprofilespatialenv.yml

