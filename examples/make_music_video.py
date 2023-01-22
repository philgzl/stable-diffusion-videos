import torch

from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers import DPMSolverMultistepScheduler


pipe = StableDiffusionWalkPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    revision="fp16",
    feature_extractor=None,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Seconds in the song.
audio_offsets = [30, 60]  # [Start, end]
fps = 5  # Use lower values for testing (5 or 10), higher values for better quality (30 or 60)

# Convert seconds to frames
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

video_path = pipe.walk(
    prompts=[
        'steampunk fractal',
        'mandelbulb acropolis',
    ],
    seeds=[123, 456],
    num_interpolation_steps=num_interpolation_steps,
    audio_filepath='Long Island Sound - I Still Love You.mp3',
    audio_start_sec=audio_offsets[0],
    fps=fps,
    batch_size=12,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dreams',        # Where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
)
