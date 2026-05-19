#!/usr/bin/env bash
# Build and launch the magician_grabber_annotator Docker container.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPOSITORY="$( cd "$DIR/.." && pwd )"

NAME="magician_grabber_annotator"

export DOCKER_BUILDKIT=1

docker pull tensorflow/tensorflow:latest-gpu

# Build context is the project root so the Dockerfile can COPY requirements.txt
docker build \
    --ssh default \
    -t $NAME \
    -f "$DIR/Dockerfile" \
    "$REPOSITORY" \
    --build-arg user_id=$UID

# Remove a stale container with the same name if it exists
docker rm -f ${NAME}-container 2>/dev/null || true

xhost +local:docker

# --network host  : ROS2 DDS (multicast/unicast) works transparently with the host
# --ipc host      : shared memory segments (SharedMemoryManager) are shared with the host
docker run -d \
    --gpus all \
    --shm-size 32G \
    --cap-add=SYS_NICE \
    --network host \
    --ipc host \
    --mount type=tmpfs,destination=/home/user/ram,tmpfs-mode=1777 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -it \
    --name ${NAME}-container \
    -v "$REPOSITORY":/home/user/workspace \
    -v /storage:/storage \
    $NAME

docker ps -a

OUR_DOCKER_ID=$(docker ps -a | grep $NAME | head -1 | cut -f1 -d' ')
echo "Container ID: $OUR_DOCKER_ID"
echo "To monitor resource usage: docker stats ${NAME}-container"
echo "Attaching..."
docker attach $OUR_DOCKER_ID

exit 0
