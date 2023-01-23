import argparse

import torch
import yaml

from diffusers import DPMSolverMultistepScheduler
from stable_diffusion_videos import StableDiffusionWalkPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input configuration file')
    args = parser.parse_args()

    with open(args.input) as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg)

    pipe = StableDiffusionWalkPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        revision="fp16",
        feature_extractor=None,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    num_interpolation_steps = [
        (b-a)*cfg.fps for a, b in zip(cfg.audio_offsets, cfg.audio_offsets[1:])
    ]

    pipe.walk(
        prompts=cfg.prompts,
        seeds=cfg.seeds,
        num_interpolation_steps=num_interpolation_steps,
        audio_filepath=cfg.audio_filepath,
        audio_start_sec=cfg.audio_offsets[0],
        fps=cfg.fps,
        batch_size=cfg.batch_size,
        height=cfg.height,
        width=cfg.width,
        output_dir=cfg.output_dir,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
    )


if __name__ == '__main__':
    main()
