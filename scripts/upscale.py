import argparse
import os
import json
import logging

from stable_diffusion_videos import RealESRGANModel, make_video_pyav


def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=lambda x: x.rstrip("/\\"),
                        help='input dream dir')
    parser.add_argument('--image-file-ext', nargs=1, default='.png')
    args = parser.parse_args()

    with open(os.path.join(args.input, 'prompt_config.json')) as f:
        cfg = json.load(f)
    cfg = argparse.Namespace(**cfg)

    output_path = f'{args.input}-upscaled'

    model = RealESRGANModel.from_pretrained('nateraw/real-esrgan')
    model.upsample_imagefolder(args.input, output_path)

    name = os.path.basename(args.input)
    output_filepath = f'{output_path}/{name}.mp4'

    make_video_pyav(
        output_path,
        audio_filepath=cfg.audio_filepath,
        fps=cfg.fps,
        audio_offset=cfg.audio_start_sec,
        audio_duration=sum(cfg.num_interpolation_steps)/cfg.fps,
        output_filepath=output_filepath,
        glob_pattern=f"**/*{args.image_file_ext}",
        sr=44100,
    )


if __name__ == '__main__':
    main()
