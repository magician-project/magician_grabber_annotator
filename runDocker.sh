export DISPLAY=$(echo $DISPLAY)
xhost +

#Ask the image name
echo "Enter the Docker image name (default: annotator):"
read -r IMAGE_NAME
IMAGE_NAME=${IMAGE_NAME:-annotator}

docker run -it --rm \
    --ipc=host \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/app \
    "$IMAGE_NAME" \
    bash