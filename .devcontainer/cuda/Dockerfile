FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Install TensorRT
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10="*+cuda${CUDA_VERSION%.*}" \
    libnvonnxparsers10="*+cuda${CUDA_VERSION%.*}" \
    && apt-mark hold \
    libnvinfer10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*
