#!/bin/sh
#
#SBATCH --job-name="avdr-RP-train-all"
#SBATCH --partition=memory
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=64G

# may want to comment this out if unneccesary
mkdir -p /scratch/${USER}/.local
ln -s /scratch/${USER}/.local $HOME/.local
mkdir -p /scratch/${USER}/.cache
ln -s /scratch/${USER}/.cache $HOME/.cache

module load 2022r1
module load compute
module load python
module load py-pip/21.1.2-v5263jz

# module load py-numpy/1.19.5-h6zz3af
# module load py-scikit-learn/1.0.1-kq4shxm
# # module load click unneccesary for this
# module load py-scipy/1.5.2-fbvrlx7
# module load py-tqdm/4.62.3-gby5l2e

python -m pip install --user -r requirements.txt

srun python recommender/classify_all.py full