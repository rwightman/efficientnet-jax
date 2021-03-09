FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /workspace

RUN pip install --upgrade pip

RUN git clone https://github.com/google/jax &&\
    cd jax &&\
    python build/build.py --enable_cuda &&\
    pip install dist/*.whl &&\
    pip install -e . &&\
    rm -rf /root/.cache/bazel && \
    cd ..

RUN git clone https://github.com/google/flax &&\
    cd flax &&\
    pip install -e . &&\
    cd ..

RUN git clone https://github.com/google/objax &&\
    cd objax &&\
    pip install -e . &&\
    cd ..

# install timm for PyTorch data pipeline / helpers that I'm familiar with, reinstall SIMD Pillow since
# it never stays installed due to dep issues
RUN pip install timm &&\
    pip uninstall -y pillow &&\
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd