#!/bin/bash

# Allow local X server connections
xhost +local:docker

docker run -it --rm \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $(pwd):/root/starkit-ransac \
    --device /dev/dri:/dev/dri \
    open3d-starkit-ransac

# Revoke X server access when done
xhost -local:docker
