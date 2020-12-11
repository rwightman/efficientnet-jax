FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

WORKDIR /workspace

RUN pip install --upgrade pip

# Default in build.py script
# ENV CUDA_COMPUTE="3.5,5.2,6.0,6.1,7.0"

# Pascal, Volta, Turing
ENV CUDA_COMPUTE="6.1,7.0,7.5"

# If you're lucky and you know it... Ampere (A100, RTX 3000)
# ENV CUDA_COMPUTE="8.0,8.6"

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


