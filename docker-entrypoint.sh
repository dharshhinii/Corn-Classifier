#!/bin/sh
set -eu

MODEL_NAME_ARG="${1:-mobilenet_finetuned}"

case "$MODEL_NAME_ARG" in
  baseline|mobilenet_finetuned|light)
    shift || true
    exec python /app/run.py \
      --model "$MODEL_NAME_ARG" \
      --host 0.0.0.0 \
      --port "${PORT:-8000}" \
      "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
