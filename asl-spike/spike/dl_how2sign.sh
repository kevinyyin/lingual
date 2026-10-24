#!/usr/bin/env bash
# How2Sign (CC BY-NC 4.0): frontal sentence clips + realigned English translations, val and test only.
set -u
cd "$(dirname "$0")/../data/how2sign"
g() { uv run python -m gdown "$1" -O "$2" || echo "FAILED $2"; }
g 1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI how2sign_realigned_val.csv
g 1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu how2sign_realigned_test.csv
g 1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO val_rgb_front_clips.zip
g 1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG test_rgb_front_clips.zip
ls -la
