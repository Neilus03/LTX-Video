#!/bin/bash
#SBATCH --job-name=ltx_i2v_flowerbed
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

#################### prompts & difficulty ####################
PROMPTS=(
"The flowers gently sway in the breeze."
"The petals of a flower slowly flutter as the wind passes."
"The leaves on the plants move slightly with the light wind."
"A single flower’s stem bends slightly as the breeze blows."
"Petals on flowers gently open and close as if reacting to a light wind."
"The flowers sway together in rhythm with a moderate breeze."
"Petals flutter in the breeze, with flowers bending gently as the wind changes direction."
"Time-lapse of flowers blooming and petals gently falling off."
"The flowers tilt as the wind pushes them, creating dynamic shifting movements."
"Leaves flutter and flowers bend as wind gusts come and go."
"The flowers morph into large, colorful butterflies that take flight around the garden."
"The flowerbed transforms, with flowers turning into floating, glowing orbs that sway in the air."
"Flowers shift into abstract geometric shapes, swirling and growing larger with the wind."
"The flowerbed twists into a magical garden where the flowers grow and shrink rapidly, changing colors with each breeze."
"The plants and flowers rearrange themselves, moving like robotic arms that stretch and grow into new shapes."
)

DIFFS=( easy easy easy easy easy
        medium medium medium medium medium
        hard hard hard hard hard )

PROMPT="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"
DIFF="${DIFFS[$SLURM_ARRAY_TASK_ID]}"

#################### optional seed flag ####################
SEED=17
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed) [[ $2 == rand ]] && SEED=$RANDOM || SEED=$2; shift 2;;
    *) shift;;
  esac
done

#################### env, paths, log ####################
eval "$(/storage/user/deln/miniconda3/bin/conda shell.bash hook)"
conda activate ltx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATE_DIR=$(date +%Y-%m-%d)
PROJECT_DIR=$PWD
IMG_PATH=$PROJECT_DIR/images/flowerbed.png
RUNID=$(date +%Y%m%d_%H%M%S)
OUTDIR=$PROJECT_DIR/outputs/flowerbed/${DIFF}/${DATE_DIR}/${RUNID}-${SEED}
mkdir -p "$OUTDIR"

LOG_DIR="$PROJECT_DIR/slurm/logs/flowerbed/${DATE_DIR}"
mkdir -p "$LOG_DIR"
mv "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/ltx_i2v_flowerbed-%A_%a.out" 2>/dev/null || true
mv "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/ltx_i2v_flowerbed-%A_%a.err" 2>/dev/null || true

#################### run ####################
echo "Prompt: $PROMPT"
echo "Difficulty: $DIFF  Seed: $SEED  Output: $OUTDIR"
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
