#!/bin/bash

DIR="$(cd -P "$(dirname "$0")" && pwd)"
# docker run --gpus all -v "$DIR":/app/workdir --network=host -ti project-tfod:1.6 bash

docker run --gpus all \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  -v "$DIR":/app/workdir \
  -v "/media/rutvik/New Volume4/Workspace/DL Project/Obj Detection/nuScenes/samples":/app/workdir/dataset --network=host -ti project-tfod:1.6 bash
    # line added only for EDA, delte it later.
# --runtime=nvidia --privileged

