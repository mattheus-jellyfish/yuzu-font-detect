FROM docker.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY requirements.txt $HOME/app/requirements.txt
RUN pip install -r requirements.txt

USER root
# Download model and font cache
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import hf_hub_download; \
    model_path = hf_hub_download('gyrojeff/YuzuMarker.FontDetection', '4x-epoch=18-step=368676.ckpt'); \
    cache_path = hf_hub_download('gyrojeff/YuzuMarker.FontDetection', 'font_demo_cache.bin'); \
    import shutil; \
    shutil.copy(cache_path, '$HOME/app/font_demo_cache.bin'); \
    shutil.copy(model_path, '$HOME/app/model.ckpt')"

# Generate font demo images if needed
RUN mkdir -p $HOME/app/demo_fonts

USER user
COPY --chown=user detector $HOME/app/detector
COPY --chown=user font_dataset $HOME/app/font_dataset
COPY --chown=user utils $HOME/app/utils
COPY --chown=user configs $HOME/app/configs
COPY --chown=user demo.py $HOME/app/demo.py

# Set environment variable for CUDA optimization
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Run with L4 optimizations
CMD ["python", "demo.py", "-d", "0", "-c", "model.ckpt", "-m", "resnet50", "-z", "512", "-p", "7860", "-a", "0.0.0.0"]
