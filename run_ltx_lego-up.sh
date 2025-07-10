#!/bin/bash
#SBATCH --job-name=ltx_i2v_lego-up
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=neil.de@tum.de
#SBATCH --constraint="GPU_CC:8.9"
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Create date-based log directory
DATE_DIR=$(date +%Y-%m-%d)
LOG_DIR="/home/stud/deln/storage/user/projects/LTX-Video/slurm/logs/lego-up/${DATE_DIR}"
mkdir -p "$LOG_DIR"

# Move SLURM output files to the log directory
mv "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/ltx_i2v_lego-up-${SLURM_JOB_ID}.out"
mv "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/ltx_i2v_lego-up-${SLURM_JOB_ID}.err"

# Parse command line arguments
SEED=17  # Default seed
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      if [[ "$2" == "rand" ]]; then
        SEED=$RANDOM
        echo "Using random seed: $SEED"
      else
        SEED="$2"
      fi
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# --- Conda env ---
eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"
conda activate ltx

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
date

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# paths
PROJECT_DIR=$PWD
IMG_PATH=$PROJECT_DIR/images/lego-up.png
PROMPT="toy lego bulldozer lifts up its front shovel"
TAG=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_')

RUNID=$(date +%Y%m%d_%H%M%S)
OUTDIR=$PROJECT_DIR/outputs/lego-up/${DATE_DIR}/$RUNID-$SEED
mkdir -p "$OUTDIR"

# ----------------- Run -----------------
echo "Starting execution at: $(date)"
start_time=$(date +%s)

srun time -v python inference.py \
    --prompt "$PROMPT" \
    --conditioning_media_paths "$IMG_PATH" \
    --conditioning_start_frames 0 \
    --height 320 \
    --width 512 \
    --num_frames 32 \
    --seed "$SEED" \
    --pipeline_config configs/ltxv-2b-0.9.6-distilled.yaml \
    --output_path "$OUTDIR" \
    --device cuda:0 \
    --offload_to_cpu

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Finished execution at: $(date)"
echo "Total execution time: ${duration} seconds ($(date -u -d @${duration} +'%H:%M:%S'))"