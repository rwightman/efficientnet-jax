FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

WORKDIR /workspace

RUN git clone https://github.com/google/jax &&\
    cd jax &&\
    python build/build.py --enable_cuda &&\
    pip install -e build &&\
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



