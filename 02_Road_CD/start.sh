#!/bin/bash

DIR=`dirname "$0"`
docker build -t opencd_pytorch:latest .
# docker load -i $DIR/opencd_pytorch.tar
docker-compose up
# docker run -it -v $DIR/workspace:/workspace --gpus=all --ipc=host opencd_pytorch:latest