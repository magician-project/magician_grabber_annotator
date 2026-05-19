#!/bin/bash
# Install ROS2 Rolling and colcon. Runs as root (called from Dockerfile).
set -e

apt-get install -y --no-install-recommends locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

apt-get install -y --no-install-recommends \
    software-properties-common \
    curl

add-apt-repository universe
apt-get update

# ROS2 apt source (detects Ubuntu codename automatically)
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
apt-get install -y /tmp/ros2-apt-source.deb
rm /tmp/ros2-apt-source.deb

apt-get update && apt-get install -y --no-install-recommends \
    ros-dev-tools \
    ros-rolling-ros-base \
    python3-colcon-common-extensions

# Extra Python packages required to use ROS inside a venv
python3 -m pip install -U \
    empy \
    lark \
    "pytest>=5.3" \
    pytest-repeat \
    pytest-rerunfailures

apt-get clean && rm -rf /var/lib/apt/lists/*
