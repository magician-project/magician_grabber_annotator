#!/bin/bash
# Container entrypoint: source ROS, build custom messages, activate venv.
set -e

source /opt/ros/rolling/setup.bash

ROS_WS="/home/${USER}/ros_ws"
WORKSPACE="/home/${USER}/workspace"

# Build the magician_vision_classifier ROS package (msgs/srvs) if not already built.
# Must happen here because the package lives in the volume-mounted workspace.
if [ ! -f "${ROS_WS}/install/setup.bash" ]; then
    echo "[entrypoint] Building ROS workspace..."
    mkdir -p "${ROS_WS}/src"
    ln -sfn "${WORKSPACE}" "${ROS_WS}/src/magician_grabber_annotator"
    cd "${ROS_WS}"
    colcon build --symlink-install
    echo "[entrypoint] Build complete."
fi

source "${ROS_WS}/install/setup.bash"

source "/home/${USER}/venv/bin/activate"

exec "$@"
