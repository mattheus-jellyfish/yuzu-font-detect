FROM docker.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt

COPY ./demo_fonts /workspace/demo_fonts
COPY ./font_demo_cache.bin /workspace/font_demo_cache.bin
