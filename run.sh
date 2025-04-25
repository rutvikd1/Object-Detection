#!/bin/bash

DIR="$(cd -P "$(dirname "$0")" && pwd)"

# docker run --gpus all -v "$DIR":/app --network=host -ti project-tfod:1.0 bash
docker run --gpus all -v "$DIR":/app/workdir --network=host -ti project-tfod:1.0 bash
