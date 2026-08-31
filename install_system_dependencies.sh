#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo: sudo bash $0" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  software-properties-common curl gnupg lsb-release locales ca-certificates

add-apt-repository -y universe

install -d -m 0755 /usr/share/keyrings
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

architecture="$(dpkg --print-architecture)"
codename="$(. /etc/os-release && printf '%s' "${UBUNTU_CODENAME}")"
printf 'deb [arch=%s signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu %s main\n' \
  "${architecture}" "${codename}" > /etc/apt/sources.list.d/ros2.list

apt-get update
apt-get install -y \
  ros-humble-desktop \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  cmake \
  ninja-build \
  git \
  git-lfs \
  libboost-all-dev \
  libeigen3-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libopencv-dev \
  libpcl-dev \
  libzmq3-dev \
  libyaml-cpp-dev \
  nlohmann-json3-dev \
  pybind11-dev \
  libtbb-dev \
  libopenblas-dev \
  liblapack-dev \
  tensorrt \
  python3-libnvinfer \
  python3-libnvinfer-dev

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  rosdep init
fi

echo "System and ROS dependencies installed successfully."
