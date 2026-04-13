#!/bin/bash
set -eux

cd /home/ec2-user

if [ ! -d system2-neural-inference ]; then
  git clone https://github.com/anna-d/system2-neural-inference.git
fi

cd /home/ec2-user/system2-neural-inference

git fetch origin
git reset --hard origin/main

source /opt/pytorch/bin/activate || true
pip install -r requirements.txt

chown -R ec2-user:ec2-user /home/ec2-user/system2-neural-inference