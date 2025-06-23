#!/bin/bash
#SBATCH --job-name=ltx_i2v_bear
#SBATCH --array=0-14            # 15 tasks → 0-14
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=neil.de@tum.de
#SBATCH --constraint="GPU_CC:8.9"
#SBATCH --output=slurm-%A_%a.out   # %A=array master ID, %a=task index
#SBATCH --error=slurm-%A_%a.err

####################### 1 PROMPTS & DIFFICULTIES #########################
PROMPTS=(
"The trees around the bear statue sway gently with the wind."
"Leaves fall slowly around the bear statue as the wind picks up."
"The bear statue stays still as the forest plants gently sway in the breeze."
"A soft breeze moves the trees around the bear statue, causing subtle leaf movement."
"The surrounding plants rustle in the wind, while the bear statue remains unmoved."
"The trees around the bear statue sway dramatically as strong gusts of wind pass through."
"The bear statue stays still as the surrounding trees and plants shake violently from the wind."
"The bear statue stands firm as the surrounding forest plants sway and twist with the wind."
"The camera circles the bear statue while the trees around it bend under the pressure of the wind."
"The bear statue is illuminated by shifting sunlight, while the forest around it reacts to gusts of wind."
"The bear statue morphs into a roaring bear that comes to life, growling and moving through the forest."
"The bear statue breaks free from its stone base and transforms into a giant stone creature, walking through the forest."
"The bear statue disintegrates and reforms into a large bear made of vines and roots, walking around the forest."
"The stone bear statue turns into a mechanical bear, with metal plates shifting and transforming as it comes to life."
"The bear statue transforms into a giant living bear that stomps through the forest, shaking the ground beneath it."
)

DIFFS=(
"easy"  "easy"  "easy"  "easy"  "easy"
"medium" "medium" "medium" "medium" "medium"
"hard"  "hard"  "hard"  "hard"  "hard"
)

PROMPT="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"
DIFF="${DIFFS[$SLURM_ARRAY_TASK_ID]}"

####################### 2 OPTIONAL SEED FLAG #############################
SEED=17
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
    *) shift ;;
  esac
done

####################### 3 ENVIRONMENT ###################################
eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"
conda activate ltx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

####################### 4 PATHS & LOG ###################################
DATE_DIR=$(date +%Y-%m-%d)
PROJECT_DIR=$PWD
IMG_PATH=$PROJECT_DIR/images/bear.png
RUNID=$(date +%Y%m%d_%H%M%S)
OUTDIR=$PROJECT_DIR/outputs/bear/${DIFF}/${DATE_DIR}/${RUNID}-${SEED}
mkdir -p "$OUTDIR"

# move slurm files to dated log dir (safe because fd follows move)
LOG_DIR="$PROJECT_DIR/slurm/logs/bear/${DATE_DIR}"
mkdir -p "$LOG_DIR"
mv "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/ltx_i2v_bear-%A_%a.out" 2>/dev/null || true
mv "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/ltx_i2v_bear-%A_%a.err" 2>/dev/null || true

####################### 5 RUN ###########################################
echo "Prompt: $PROMPT"
echo "Difficulty: $DIFF"
echo "Seed: $SEED"
echo "Output dir: $OUTDIR"
start_time=$(date +%s)

srun time -v python inference.py \
  --prompt "$PROMPT" \
  --conditioning_media_paths "$IMG_PATH" \
  --conditioning_start_frames 0 \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --seed "$SEED" \
  --pipeline_config configs/ltxv-2b-0.9.6-distilled.yaml \
  --output_path "$OUTDIR" \
  --device cuda:0 \
  --offload_to_cpu

duration=$(( $(date +%s) - start_time ))
echo "Finished in ${duration} s"
