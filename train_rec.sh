#!/bin/sh

# may want to comment this out if unneccesary
mkdir -p /scratch/${USER}/.local
ln -s /scratch/${USER}/.local $HOME/.local
mkdir -p /scratch/${USER}/.cache
ln -s /scratch/${USER}/.cache $HOME/.cache

module load 2022r1
module load compute
module load python
module load py-pip/21.1.2-v5263jz

python -m pip install --user -r requirements.txt

for hier_idx in 0 1 2 3
    for rec_idx in 0 1 2 3 4 5
        sh ./run_cluster_full.sh hier_idx rec_idx