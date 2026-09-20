#!/bin/bash
#SBATCH --job-name=nlp-p1-enc
#SBATCH --partition=lenovo
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/logs/slurm_%x_%j.out
#SBATCH --error=artifacts/logs/slurm_%x_%j.out

set -e
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8

PY=$HOME/.conda/envs/nlp-p1/bin/python
export PYTHONPATH=$PWD/src
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "node $(hostname)  date $(date)"
echo "########## keep an encoder for every dataset, then re-export the demo data ##########"
$PY src/run.py --mode main
$PY src/export_embeddings.py
echo "done $(date)"
