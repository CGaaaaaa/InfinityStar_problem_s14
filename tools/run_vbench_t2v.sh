#!/bin/bash
# VBench T2V Evaluation Pipeline for InfinityStar
# This script generates videos from VBench prompts and evaluates them

set -e

# Configuration
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $0))"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/yekeming/pretrained/InfinityStar}"
VBENCH_DIR="${PROJECT_DIR}/VBench"
OUTPUT_DIR="${PROJECT_DIR}/vbench_output"
EVAL_DIR="${PROJECT_DIR}/vbench_evaluation_results"

# Create directories
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

# Parse arguments
DURATION="${DURATION:-5}"
NUM_VIDEOS="${NUM_VIDEOS:-5}"
DIMENSIONS="${DIMENSIONS:-}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --generate         Generate videos only"
    echo "  --evaluate        Evaluate existing videos"
    echo "  --all             Generate and evaluate (default)"
    echo "  --duration N      Video duration in seconds (default: 5)"
    echo "  --num-videos N    Number of videos per prompt (default: 5)"
    echo "  --dim DIMENSIONS  Space-separated list of dimensions"
    echo "  --max-prompts N   Limit number of prompts for testing"
    echo ""
    echo "Environment variables:"
    echo "  CHECKPOINTS_DIR   Path to InfinityStar checkpoints"
    echo ""
    echo "Examples:"
    echo "  $0 --generate --duration 5"
    echo "  $0 --evaluate --dim subject_consistency temporal_flickering"
    echo "  $0 --all --num-videos 5 --max-prompts 10"
}

# Parse command line arguments
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate) MODE="generate"; shift ;;
        --evaluate) MODE="evaluate"; shift ;;
        --all) MODE="all"; shift ;;
        --duration) DURATION="$2"; shift 2 ;;
        --num-videos) NUM_VIDEOS="$2"; shift 2 ;;
        --dim) shift; DIMENSIONS="$@"; break ;;
        --max-prompts) MAX_PROMPTS="--max_prompts $2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

echo "=========================================="
echo "VBench T2V Evaluation Pipeline"
echo "=========================================="
echo "Mode: $MODE"
echo "Duration: ${DURATION}s"
echo "Videos per prompt: $NUM_VIDEOS"
echo "Output dir: $OUTPUT_DIR"
echo "Eval dir: $EVAL_DIR"
echo "=========================================="

# Step 1: Generate videos
if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
    echo ""
    echo "[1/2] Generating videos from VBench prompts..."
    echo "-------------------------------------------"

    CMD="python $SCRIPT_DIR/infer_vbench_t2v.py \
        --vbench_prompt_json $PROJECT_DIR/evaluation/VBench_rewrited_prompt.json \
        --output_dir $OUTPUT_DIR \
        --checkpoints_dir $CHECKPOINTS_DIR \
        --duration $DURATION \
        --num_videos_per_prompt $NUM_VIDEOS \
        --fps 16"

    if [[ -n "$DIMENSIONS" ]]; then
        CMD="$CMD --dimensions $DIMENSIONS"
    fi

    if [[ -n "$MAX_PROMPTS" ]]; then
        CMD="$CMD $MAX_PROMPTS"
    fi

    eval $CMD

    echo "Video generation complete!"
fi

# Step 2: Evaluate videos
if [[ "$MODE" == "evaluate" || "$MODE" == "all" ]]; then
    echo ""
    echo "[2/2] Evaluating generated videos..."
    echo "-------------------------------------------"

    # Default dimensions for evaluation
    EVAL_DIMS="subject_consistency background_consistency temporal_flickering \
               motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
               object_class multiple_objects human_action color \
               spatial_relationship scene temporal_style appearance_style overall_consistency"

    for dim in $EVAL_DIMS; do
        echo "Evaluating: $dim"
        python $SCRIPT_DIR/evaluate_vbench.py \
            --videos_path "$OUTPUT_DIR/$dim" \
            --output_path "$EVAL_DIR" \
            --full_json_dir "$VBENCH_DIR/vbench/VBench_full_info.json" \
            --dimension "$dim" \
            --mode custom_input || true
    done

    echo "Evaluation complete!"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
