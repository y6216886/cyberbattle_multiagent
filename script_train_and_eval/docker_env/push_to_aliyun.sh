#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./push_to_aliyun.sh <aliyun-registry> <namespace> <repo> <tag>
# Example:
#   ./push_to_aliyun.sh registry.cn-hangzhou.aliyuncs.com mynamespace myenv image-v1
# The script will prompt for username/password (or you can set DOCKER_USERNAME/DOCKER_PASSWORD env vars).

REGISTRY="$1"
NAMESPACE="$2"
REPO="$3"
TAG="$4"

IMAGE_LOCAL="${REPO}:temp-build"
IMAGE_REMOTE="${REGISTRY}/${NAMESPACE}/${REPO}:${TAG}"

# Ensure requirements.txt exists at repo root
ROOT_REQUIREMENTS="$(pwd)/requirements.txt"
if [ ! -f "$ROOT_REQUIREMENTS" ]; then
  echo "requirements.txt not found at repository root: $ROOT_REQUIREMENTS"
  exit 1
fi

# Create temp build context and copy only what we need (requirements + Dockerfile)
TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

cp "$(pwd)/script_train_and_eval/docker_env/Dockerfile.env" "$TMPDIR/Dockerfile"
cp "$ROOT_REQUIREMENTS" "$TMPDIR/requirements.txt"

echo "Building Docker image (environment-only) locally..."
docker build -t "$IMAGE_LOCAL" "$TMPDIR"

echo "Tagging image for remote: $IMAGE_REMOTE"
docker tag "$IMAGE_LOCAL" "$IMAGE_REMOTE"

# Login (ask for credentials if not provided)
if [ -z "${DOCKER_USERNAME:-}" ]; then
  read -p "Docker registry username: " DOCKER_USERNAME
fi
if [ -z "${DOCKER_PASSWORD:-}" ]; then
  read -s -p "Docker registry password (or token): " DOCKER_PASSWORD
  echo
fi

echo "Logging in to $REGISTRY..."
echo "$DOCKER_PASSWORD" | docker login "$REGISTRY" -u "$DOCKER_USERNAME" --password-stdin

echo "Pushing image to $IMAGE_REMOTE"
docker push "$IMAGE_REMOTE"

echo "Image pushed successfully: $IMAGE_REMOTE"

echo "You can inspect the environment metadata by running:\n  docker run --rm $IMAGE_REMOTE cat /opt/env/env_info.json"

# Optionally remove local temporary tag
# docker rmi "$IMAGE_LOCAL" || true

exit 0
