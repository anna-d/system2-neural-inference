#!/bin/bash
exec > >(tee /var/log/system2-bootstrap.log | logger -t user-data ) 2>&1
set -exo pipefail

REPO_DIR="/home/ec2-user/system2-neural-inference"
REPO_URL="https://github.com/anna-d/system2-neural-inference.git"

cd /home/ec2-user

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git reset --hard origin/main

source /opt/pytorch/bin/activate || true

if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

mkdir -p artifacts results logs
chown -R ec2-user:ec2-user "$REPO_DIR"

python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "Bootstrap complete"