#!/bin/bash
#SBATCH --job-name=ltx_i2v_lego-up
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
"The bulldozer moves slowly forward, pushing small Lego pieces as its bucket lifts them."
"The bulldozer's bucket tilts up and down as it moves across the table."
"The bulldozer rolls across the table, with its tracks spinning gently."
"The Lego bulldozer’s bucket scoops Lego pieces and moves them slightly."
"The bulldozer moves forward, its tracks spinning smoothly and pushing Lego pieces in front of it."
"The bulldozer turns sharply, its tracks shifting direction as the bucket moves up and down."
"The bulldozer moves in a circular pattern, pushing Lego bricks with its bucket and lifting them up."
"The bulldozer picks up and drops Lego pieces in a rhythmic motion as it moves around the table."
"The bulldozer moves rapidly in different directions, its bucket lifting and dumping Lego pieces as it works."
"The bulldozer scoops up a pile of Lego bricks and stacks them in a corner of the table."
"The Lego bulldozer transforms into a humanoid-robot, standing up and walking around the table."
"The bulldozer’s tracks and bucket transform into robotic arms and legs, and it begins moving like a large robot."
"The bulldozer turns into a flying vehicle, with tracks turning into propellers as it hovers above the table."
"The Lego bulldozer transforms into a futuristic construction robot, with its arms and legs extending into a massive robotic form."
"The bulldozer morphs into a fully functional robot, transforming its bucket into a large laser cutter that scans and analyzes Lego pieces."
)
DIFFS=( easy easy easy easy easy
        medium medium medium medium medium
        hard hard hard hard hard )

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
IMG_PATH=$PROJECT_DIR/images/lego-up.png     # keep original filename
RUNID=$(date +%Y%m%d_%H%M%S)
DIFF="${DIFFS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR=$PROJECT_DIR/outputs/lego-up/${DIFF}/${DATE_DIR}/${RUNID}-${SEED}
mkdir -p "$OUTDIR"

LOG_DIR="$PROJECT_DIR/slurm/logs/lego-up/${DATE_DIR}"
mkdir -p "$LOG_DIR"
mv "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/ltx_i2v_lego-up-%A_%a.out" 2>/dev/null || true
mv "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/ltx_i2v_lego-up-%A_%a.err" 2>/dev/null || true

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
