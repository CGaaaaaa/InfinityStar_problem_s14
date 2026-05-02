"""
VBench T2V Generation Script for InfinityStar

This script generates videos from VBench prompts to evaluate InfinityStar's performance.
It follows the VBench evaluation protocol:
- Each prompt generates 5 videos (index 0-4)
- Videos are named as {prompt}-{index}.mp4
- Random seed is set for reproducibility
"""

import sys
import json
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import torch
import argparse
from PIL import Image
import subprocess
import glob

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args


def setup_model(args):
    """Initialize the InfinityStar model pipeline."""
    from tools.prompt_rewriter import OpenAIGPTModel

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    vae = vae.float().to('cuda')
    # load infinity
    infinity = load_transformer(vae, args)
    self_correction = SelfCorrection(vae, args)

    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

    return {
        'text_tokenizer': text_tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'infinity': infinity,
        'self_correction': self_correction,
        'video_encode': video_encode,
        'video_decode': video_decode,
        'get_visual_rope_embeds': get_visual_rope_embeds,
        'get_scale_pack_info': get_scale_pack_info,
    }


def generate_video(pipe, args, prompt, seed, duration, num_videos=1, output_dir=None, video_name_prefix=None):
    """
    Generate video(s) from a text prompt.

    Args:
        pipe: Model pipeline
        args: Model arguments
        prompt: Text prompt for video generation (refined prompt)
        seed: Random seed
        duration: Video duration in seconds
        num_videos: Number of videos to generate per prompt
        output_dir: Output directory for videos
        video_name_prefix: Original short prompt for filename (VBench standard)

    Returns:
        List of generated video paths
    """
    num_frames = duration * 16 + 1
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe['get_scale_pack_info'](scale_schedule, args.first_full_spatial_size_scale_index, args)
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16

    generated_video_paths = []

    for video_idx in range(num_videos):
        # Set random seed for each video
        torch.manual_seed(seed + video_idx)
        np.random.seed(seed + video_idx)

        # Text-to-Video (no image conditioning)
        gt_leak, gt_ls_Bl = -1, None

        prompt_with_suffix = f'{prompt}, Close-up on big objects, emphasize scale and detail'
        if args.append_duration2caption:
            prompt_with_suffix = f'<<<t={duration}s>>>' + prompt_with_suffix

        negative_prompt = ""

        start_time = time.time()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            generated_image, _ = gen_one_example(
                pipe['infinity'],
                pipe['vae'],
                pipe['text_tokenizer'],
                pipe['text_encoder'],
                prompt_with_suffix,
                negative_prompt=negative_prompt,
                g_seed=seed + video_idx,
                gt_leak=gt_leak,
                gt_ls_Bl=gt_ls_Bl,
                cfg_list=args.cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[0],
                vae_type=args.vae_type,
                sampling_per_bits=1,
                enable_positive_prompt=0,
                low_vram_mode=True,
                args=args,
                get_visual_rope_embeds=pipe['get_visual_rope_embeds'],
                context_info=context_info,
                noise_list=None,
            )

            if len(generated_image.shape) == 3:
                generated_image = generated_image.unsqueeze(0)

        # Save video - use short prompt for filename (VBench standard)
        if video_name_prefix:
            short_prompt = sanitize_filename(video_name_prefix)
        else:
            short_prompt = sanitize_filename(prompt)
        
        video_name = f'{short_prompt}-{video_idx}.mp4'
        video_path = osp.join(output_dir, video_name)

        video_output = generated_image.cpu().numpy()
        save_video(video_output, fps=args.fps, save_filepath=video_path)

        elapsed_time = time.time() - start_time
        print(f"  Generated video {video_idx + 1}/{num_videos} in {elapsed_time:.2f}s: {video_path}")

        generated_video_paths.append(video_path)

    return generated_video_paths


def sanitize_filename(prompt):
    """Sanitize prompt for use as filename."""
    # Remove or replace invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    filename = prompt
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def load_vbench_prompts(json_path):
    """Load prompts from VBench_rewrited_prompt.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    return prompts_data


def main():
    parser = argparse.ArgumentParser(description='VBench T2V Generation for InfinityStar')

    # Paths
    parser.add_argument('--vbench_prompt_json', type=str,
                        default='/data/yekeming/caijiani/projects/InfinityStar/evaluation/VBench_rewrited_prompt.json',
                        help='Path to VBench_rewrited_prompt.json')
    parser.add_argument('--output_dir', type=str,
                        default='/data/yekeming/caijiani/projects/InfinityStar/vbench_output',
                        help='Output directory for generated videos')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='/data/yekeming/pretrained/InfinityStar',
                        help='Path to InfinityStar checkpoints')

    # Generation settings
    parser.add_argument('--duration', type=int, default=5, help='Video duration in seconds (5 or 10)')
    parser.add_argument('--num_videos_per_prompt', type=int, default=5,
                        help='Number of videos to generate per prompt')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--fps', type=int, default=16, help='Video FPS')

    # Dimension filter
    parser.add_argument('--dimensions', nargs='+', type=str, default=None,
                        help='Filter prompts by dimension (e.g., subject_consistency temporal_flickering)')
    parser.add_argument('--max_prompts', type=int, default=None,
                        help='Maximum number of prompts to process (for testing)')

    # Model settings
    parser.add_argument('--pn', type=str, default='0.40M', help='Model parameter count')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model arguments
    model_args = Args()
    model_args.pn = args.pn
    model_args.fps = args.fps
    model_args.video_frames = args.duration * 16 + 1
    model_args.model_path = osp.join(args.checkpoints_dir, 'infinitystar_8b_480p_weights')
    model_args.checkpoint_type = 'torch_shard'
    model_args.vae_path = osp.join(args.checkpoints_dir, 'infinitystar_videovae.pth')
    model_args.text_encoder_ckpt = osp.join(args.checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    model_args.videovae = 10
    model_args.model_type = 'infinity_qwen8b'
    model_args.text_channels = 2048
    model_args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    model_args.bf16 = 1
    model_args.use_apg = 1
    model_args.use_cfg = 0
    model_args.cfg = 34
    model_args.tau_image = 1
    model_args.tau_video = 0.4
    model_args.apg_norm_threshold = 0.05
    model_args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    model_args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    model_args.append_duration2caption = 1
    model_args.use_two_stage_lfq = 1
    model_args.detail_scale_min_tokens = 350
    model_args.semantic_scales = 11
    model_args.max_repeat_times = 10000
    model_args.enable_rewriter = 0
    model_args.vae_type = 64  # Default from Args class, used for codebook dimension
    model_args.use_apg = 1
    model_args.apg_norm_threshold = 0.05

    print("Loading InfinityStar model...")
    pipe = setup_model(model_args)
    print("Model loaded successfully!")

    # Load prompts
    print(f"Loading prompts from {args.vbench_prompt_json}...")
    prompts_data = load_vbench_prompts(args.vbench_prompt_json)
    print(f"Loaded {len(prompts_data)} prompts")

    # Filter by dimension if specified
    if args.dimensions:
        filtered_prompts = []
        for item in prompts_data:
            if any(dim in item.get('dimension', []) for dim in args.dimensions):
                filtered_prompts.append(item)
        prompts_data = filtered_prompts
        print(f"Filtered to {len(prompts_data)} prompts for dimensions: {args.dimensions}")

    # Limit prompts if specified
    if args.max_prompts:
        prompts_data = prompts_data[:args.max_prompts]
        print(f"Limited to {len(prompts_data)} prompts for testing")

    # Generate videos
    print(f"\nStarting video generation...")
    print(f"Duration: {args.duration}s, Videos per prompt: {args.num_videos_per_prompt}")
    print(f"Total prompts: {len(prompts_data)}")

    # Generate videos
    print(f"\nStarting video generation...")
    print(f"Duration: {args.duration}s, Videos per prompt: {args.num_videos_per_prompt}")
    print(f"Total prompts: {len(prompts_data)}")
    print(f"Output directory: {args.output_dir} (all videos in single directory)")

    total_videos = 0
    start_time = time.time()

    for i, item in enumerate(prompts_data):
        prompt = item.get('prompt_en', '')  # Original short prompt (VBench standard)
        refined_prompt = item.get('refined_prompt', prompt)  # Long refined prompt for generation
        dimension = item.get('dimension', ['unknown'])[0]

        print(f"\n[{i+1}/{len(prompts_data)}] Prompt: {prompt[:80]}... [{dimension}]")

        # Generate videos using refined prompt, but name files with original prompt
        try:
            video_paths = generate_video(
                pipe=pipe,
                args=model_args,
                prompt=refined_prompt,  # Use refined prompt for generation
                seed=args.seed + i * 1000,
                duration=args.duration,
                num_videos=args.num_videos_per_prompt,
                output_dir=args.output_dir,  # All videos in single directory
                video_name_prefix=prompt  # Use original prompt for filename
            )
            total_videos += len(video_paths)
        except Exception as e:
            print(f"    ERROR generating video: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Total videos: {total_videos}")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
