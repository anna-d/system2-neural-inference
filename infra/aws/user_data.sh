#!/bin/bash
set -eux

cd /home/ec2-user

if [ ! -d system2-neural-inference ]; then
  git clone https://github.com/anna-d/system2-neural-inference.git
fi

cd system2-neural-inference

source /opt/pytorch/bin/activate || true
pip install -r requirements.txt