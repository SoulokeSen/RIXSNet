# [RIXS Spectra Using Geometric Algebra Graph Neural Networks] 


## Requirement and Conda Environment
```
conda create -n mvn python=3.10 -c conda-forge -y
conda activate mvn
conda install -c conda-forge -y numpy=2.2.5 scipy scikit-learn joblib matplotlib h5py=3.15.1 pyyaml rdkit pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip install wandb
```

Use following commands to set up the folder.
```
cd lib/engineer/
pip install -e .
cd ../../mvn
```

## Code Organization
* `mvn/`: contains the core code snippets.
  * `algebra/`: contains the Clifford Algebra implementation.
  * `configs/`: contains the configuration files. 
  * `data/`: contains necessary (simplicial) data modules.
  * `models/`: contains model and layer implementations.
* `engineer/`: contains the training and evaluation scripts.
* `lib/` contains commands and set up

## Datasets and Experiments

### Datasets
Run `mkdir ./datasets/` to generate a folder for storing datasets of experiments

#### NBody
Download / Generate nbody datasets and move to `./datasets/`

### Experiments
This implementation uses conda environment, change the path of `miniconda/` in `activate.sh` to your local `miniconda/` path and run `sh ../activate.sh`.

#### Instruction to run sweep_local for MVN on pNBody simulation task
* Clifford_EGNN: ```sweep_local configs/rixs_clifford_egnn.yaml```

