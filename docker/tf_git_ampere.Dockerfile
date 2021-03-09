FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3

WORKDIR /workspace

RUN pip install --upgrade pip

# If you're lucky and you know it... Ampere (A100, RTX 3000)
ENV CUDA_COMPUTE="8.0,8.6"

RUN git clone https://github.com/google/jax &&\
    cd jax &&\
    python build/build.py --enable_cuda --cuda_compute_capabilities=$CUDA_COMPUTE &&\
    pip install dist/*.whl &&\
    pip install -e . &&\
    rm -rf /root/.cache/bazel &&\
    cd ..

RUN git clone https://github.com/google/flax &&\
    cd flax &&\
    pip install -e . &&\
    cd ..

RUN git clone https://github.com/google/objax &&\
    cd objax &&\
    pip install -e . &&\
    cd ..

RUN pip install ml_collections


