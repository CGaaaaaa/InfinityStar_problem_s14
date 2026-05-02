#!/usr/bin/env python3
"""
VBench Evaluation Script for InfinityStar

This script evaluates generated videos using VBench evaluation framework.
It supports both standard VBench evaluation and custom prompt evaluation.
"""

import sys
import os
import os.path as osp
import argparse
from pathlib import Path

# Add VBench to path
sys.path.append(osp.join(osp.dirname(__file__), '..', 'VBench'))

def parse_args():
    parser = argparse.ArgumentParser(description='VBench Evaluation for InfinityStar')

    # Paths
    parser.add_argument('--videos_path', type=str, required=True,
                        help='Path to generated videos folder')
    parser.add_argument('--output_path', type=str,
                        default='/data/yekeming/caijiani/projects/InfinityStar/vbench_evaluation_results',
                        help='Output path for evaluation results')
    parser.add_argument('--full_json_dir', type=str,
                        default='/data/yekeming/caijiani/projects/InfinityStar/VBench/vbench/VBench_full_info.json',
                        help='Path to VBench_full_info.json')

    # Evaluation settings
    parser.add_argument('--dimension', nargs='+', type=str,
                        default=['subject_consistency', 'background_consistency', 'temporal_flickering',
                                'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
                                'object_class', 'multiple_objects', 'human_action', 'color',
                                'spatial_relationship', 'scene', 'temporal_style', 'appearance_style',
                                'overall_consistency'],
                        help='List of evaluation dimensions')
    parser.add_argument('--mode', type=str, choices=['custom_input', 'vbench_standard', 'vbench_category'],
                        default='vbench_standard',  # Use vbench_standard to match VBench official evaluation
                        help='Evaluation mode')
    parser.add_argument('--category', type=str, default=None,
                        help='Category for vbench_category mode')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs for distributed evaluation')

    # Custom input mode
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='Path to prompt JSON file (for custom_input mode)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt for evaluation (for custom_input mode)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    print(f"VBench Evaluation")
    print(f"{'='*60}")
    print(f"Videos path: {args.videos_path}")
    print(f"Output path: {args.output_path}")
    print(f"Dimensions: {args.dimension}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")

    # Import VBench
    try:
        from vbench import VBench
        import torch
        from vbench.distributed import dist_init, print0
    except ImportError as e:
        print(f"Error importing VBench: {e}")
        print("Please ensure VBench is installed: pip install vbench")
        sys.exit(1)

    # Initialize
    device = torch.device(args.device)
    my_VBench = VBench(device, args.full_json_dir, args.output_path)

    # Prepare kwargs
    kwargs = {}
    if args.category:
        kwargs['category'] = args.category

    # Evaluate each dimension
    for dim in args.dimension:
        print(f"\nEvaluating dimension: {dim}")
        print(f"{'-'*40}")

        try:
            my_VBench.evaluate(
                videos_path=args.videos_path,
                name=f'{dim}_results',
                prompt_list=[] if args.mode == 'vbench_standard' else None,
                dimension_list=[dim],
                local=True,
                mode=args.mode,
                **kwargs
            )
            print(f"Completed: {dim}")
        except Exception as e:
            print(f"Error evaluating {dim}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*60}")


def evaluate_single_dimension(videos_path, dimension, output_path, full_json_dir, mode='vbench_standard'):
    """Evaluate a single dimension."""
    import torch
    from vbench import VBench

    device = torch.device('cuda')
    my_VBench = VBench(device, full_json_dir, output_path)

    my_VBench.evaluate(
        videos_path=videos_path,
        name=f'{dimension}_results',
        prompt_list=[],
        dimension_list=[dimension],
        local=True,
        mode=mode
    )


if __name__ == '__main__':
    main()
