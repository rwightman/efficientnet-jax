FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

WORKDIR /workspace

RUN pip install --upgrade pip

ENV CUDA_VERSION=11.1

RUN pip install jaxlib && \
    pip install --upgrade -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g` && \
    pip install jax

RUN pip install objax

RUN pip install flax

