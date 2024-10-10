#!/bin/bash

DIR=`dirname "$0"`
docker load -i $DIR/opencd_pytorch.tar
docker-compose up
# docker run -it -v $DIR/workspace:/workspace --gpus=all --ipc=host opencd_pytorch:latest

# python /workspace/predict.py --output_path /workspace/out/out.gpkg
# python /workspace/predict.py --output_path /workspace/out/out.gpkg --img_1 /workspace/sample_data/A --img_2 /workspace/sample_data/B