# NVIDIA's aarch64 CUDA/PyTorch build, not the x86_64 Intel inference venv.
FROM nvcr.io/nvidia/pytorch:25.08-py3@sha256:ace9a848c0ae543317e3c4763b6b4248961c47902625abfe3c77a0fb931c50fb

ARG SAM_COMMIT=660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
RUN apt-get update && apt-get install -y --no-install-recommends \
      gstreamer1.0-tools gstreamer1.0-plugins-good libv4l-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Retain the vendor Torch/Torchvision/Triton build. Do not resolve a replacement
# PyPI torch wheel while installing the detector's pure-Python dependencies.
RUN python -m pip install --no-cache-dir --no-deps \
      numpy==1.26.4 timm==1.0.17 setuptools==78.1.0 pycocotools==2.0.11 \
      pandas==2.3.3 scipy==1.16.3 hydra-core==1.3.6 omegaconf==2.3.1 \
      psutil==7.2.2 einops==0.8.2 ftfy==6.1.1 iopath==0.1.10 \
      huggingface-hub==0.34.4 hf-xet==1.1.7 \
      portalocker==3.2.0 antlr4-python3-runtime==4.9.3 \
      dynamixel-sdk==4.0.5 pyserial==3.5 \
    && git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && git -C /opt/sam3 checkout "${SAM_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps /opt/sam3

WORKDIR /opt/turret-demo
COPY pyproject.toml README.md MANIFEST.in ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
EXPOSE 8080
ENTRYPOINT ["spring-turret"]
CMD ["--config", "/etc/spring-turret-demo.json"]
