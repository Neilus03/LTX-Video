#!/bin/bash
#SBATCH --job-name=ltx_i2v_gardenvase
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=neil.de@tum.de
#SBATCH --constraint="GPU_CC:8.9"
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

PROMPTS=(
"The dried plants in the vase sway gently with the wind."
"The plants inside the vase slightly bend with a light breeze."
"The vase remains still as the plants inside it gently move."
"A soft breeze makes the plants inside the vase shift and sway."
"The dried plants flutter gently as the breeze moves through them."
"The plants inside the vase start twisting around each other as the wind blows harder."
"The vase tilts slightly, and the plants inside it sway dramatically, some moving completely out of the vase."
"The dried plants inside the vase shift rapidly, growing longer and twisting like vines in the wind."
"The vase cracks open and plants spill out, growing new branches that sway in the breeze."
"The plants grow and twist, forming an intricate vine structure that overtakes the vase and spills out onto the table."
"The vase transforms into a tree, with the plants inside it morphing into branches and leaves that stretch upward."
"The dried plants turn into animated creatures, climbing and twisting around the vase like sentient vines."
"The vase cracks open and releases a swarm of flowers that spin and take flight around the room."
"The plants inside the vase grow rapidly, shifting into a massive tree with roots pulling the vase into the ground."
"The vase becomes a dynamic, living entity, with plants inside it growing and changing forms, shifting into different plant species as it moves."
)
DIFFS=( easy easy easy easy easy
        medium medium medium medium medium
        hard hard hard hard hard )

# seed logic identical to previous script
SEED=17
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed) [[ $2 == rand ]] && SEED=$RANDOM || SEED=$2; shift 2;;
    *) shift;;
  esac
done

eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"
conda activate ltx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATE_DIR=$(date +%Y-%m-%d)
PROJECT_DIR=$PWD
IMG_PATH=$PROJECT_DIR/images/gardenvase.png
RUNID=$(date +%Y%m%d_%H%M%S)
DIFF="${DIFFS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR=$PROJECT_DIR/outputs/gardenvase/${DIFF}/${DATE_DIR}/${RUNID}-${SEED}
mkdir -p "$OUTDIR"

LOG_DIR="$PROJECT_DIR/slurm/logs/gardenvase/${DATE_DIR}"
mkdir -p "$LOG_DIR"
mv "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/ltx_i2v_gardenvase-%A_%a.out" 2>/dev/null || true
mv "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/ltx_i2v_gardenvase-%A_%a.err" 2>/dev/null || true

PROMPT="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"
echo "Prompt: $PROMPT  Difficulty: $DIFF  Seed: $SEED"
start=$(date +%s)

srun time -v python inference.py \
  --prompt "$PROMPT" \
  --conditioning_media_paths "$IMG_PATH" \
  --conditioning_start_frames 0 \
  --height 704 --width 1216 --num_frames 121 \
  --seed "$SEED" \
  --pipeline_config configs/ltxv-2b-0.9.6-distilled.yaml \
  --output_path "$OUTDIR" \
  --device cuda:0 --offload_to_cpu

echo "Done in $(( $(date +%s) - start )) s"
