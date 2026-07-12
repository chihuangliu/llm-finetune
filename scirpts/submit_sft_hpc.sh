#!/bin/bash
# Submit a single SFT (LoRA) fine-tuning job to the HPC via PBS.
#
# Usage:
#   bash scirpts/submit_sft_hpc.sh \
#     [--output-dir DIR] [--epochs N] [--batch-size N] \
#     [--logging-steps N] [--eval-steps N]
#
# All arguments are optional and mirror the arguments of llm_finetune.sft;
# any not supplied fall back to that module's own defaults.
set -euo pipefail

PROJECT_DIR=/rds/general/user/cl3225/home/llm-finetune

# --- defaults (empty => let llm_finetune.sft use its own default) -----------
OUTPUT_DIR=""
EPOCHS=""
BATCH_SIZE=""
LOGGING_STEPS=""
EVAL_STEPS=""

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
    --epochs)        EPOCHS="$2";        shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
    --logging-steps) LOGGING_STEPS="$2"; shift 2 ;;
    --eval-steps)    EVAL_STEPS="$2";    shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# --- build the arg string forwarded to python -------------------------------
SFT_ARGS=""
[[ -n "$OUTPUT_DIR"    ]] && SFT_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$EPOCHS"        ]] && SFT_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$BATCH_SIZE"    ]] && SFT_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LOGGING_STEPS" ]] && SFT_ARGS+=" --logging-steps ${LOGGING_STEPS}"
[[ -n "$EVAL_STEPS"    ]] && SFT_ARGS+=" --eval-steps ${EVAL_STEPS}"

echo "Submitting SFT job with args:${SFT_ARGS:- (all defaults)}"

qsub -N "sft" -v SFT_ARGS="${SFT_ARGS}",PROJECT_DIR="${PROJECT_DIR}" <<'PBS'
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -q v1_gpu72

cd "${PROJECT_DIR}"

# --- logging: timestamped log/err files (PBS -o/-e can't expand variables) ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOG_BASE="$LOG_DIR/sft_${TS}_${JOBID}"
exec >"${LOG_BASE}.log" 2>"${LOG_BASE}.err"

echo "=== job ${PBS_JOBID} started at $(date) on $(hostname) ==="

source .venv/bin/activate

python -m llm_finetune.sft ${SFT_ARGS}

echo "=== job ${PBS_JOBID} finished at $(date) with exit code $? ==="
PBS

echo "Submitted SFT job."
