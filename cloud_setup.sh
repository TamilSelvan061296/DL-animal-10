#!/bin/bash

# This script sets up the environment for the DL-animal-10 project on a cloud server.

# clone my repo
git clone git@github.com:TamilSelvan061296/DL-animal-10.git

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the drivers
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# reboot the system
sudo reboot