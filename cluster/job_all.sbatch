#!/bin/bash
#SBATCH --job-name=nlp-p1
#SBATCH --partition=data-science
#SBATCH --nodelist=ds001
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=artifacts/logs/slurm_%x_%j.out
#SBATCH --error=artifacts/logs/slurm_%x_%j.out

set -e
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8

PY=$HOME/.conda/envs/nlp-p1/bin/python
export PYTHONPATH=$PWD/src
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "node $(hostname)  date $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "########## STAGE 1: main comparison ##########"
$PY src/run.py --mode main

echo "########## STAGE 2: classifier ablation on DistilBERT ##########"
$PY src/run.py --mode ablation

echo "########## STAGE 3: best ablation variant on BERT ##########"
$PY src/run.py --mode best_on_bert

echo "########## STAGE 4: embedding export for the web demo ##########"
$PY src/export_embeddings.py

echo "done $(date)"
