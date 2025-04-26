#!/bin/bash

DIR="$(cd -P "$(dirname "$0")" && pwd)"
docker run --gpus all -v "$DIR":/app/workdir --network=host -ti project-tfod:1.6 bash
