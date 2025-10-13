#!/bin/bash
# Allows the container to connect to your display for GUIs
xhost +local:root

# Run the container with maximum host resources
docker run --rm -it \
  --name h12_cameracalib-container \
  -v /home/max/vp_ws/src:/ros2_ws/src \
  --gpus all \
  --network host \
  --ipc=host \
  --pid=host \
  --privileged \
  --device /dev/bus/usb:/dev/bus/usb \
  -v /dev:/dev \
  -e DISPLAY="$DISPLAY" \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  h12_cameracalib:latest \
  "$@"
