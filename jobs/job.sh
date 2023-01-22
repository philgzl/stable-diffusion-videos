#!/bin/sh
#BSUB -q gpuv100
#BSUB -J jobname
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
export PATH=$PATH:ffmpeg-5.1.1-amd64-static
source venv/bin/activate
python scripts/make_music_video.py example_cfg.yaml
