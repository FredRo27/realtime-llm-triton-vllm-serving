#!/bin/sh
set -eu

exec tritonserver \
  --model-repository=/models \
  --model-control-mode=explicit \
  --load-model="${TRITON_MODEL_NAME:-whisper_asr}" \
  --strict-model-config=true \
  --log-verbose=0 \
  --allow-grpc=true \
  --grpc-port=8001 \
  --allow-http=true \
  --http-port=8000 \
  --allow-metrics=true \
  --metrics-port=8002
