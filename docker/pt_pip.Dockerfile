FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /workspace

RUN pip install --upgrade pip

ENV CUDA_VERSION=11.1

RUN pip install jaxlib && \
    pip install --upgrade -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g` && \
    pip install jax

RUN pip install objax

RUN pip install flax

# install timm for PyTorch data pipeline / helpers that I'm familiar with, reinstall SIMD Pillow since
# it never stays installed due to dep issues
RUN pip install timm &&\
    pip uninstall -y pillow &&\
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
