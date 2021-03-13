# Hello World!

# EfficientNet JAX - Flax Linen and Objax

## Intro
This is very much a giant steaming work in progress. Jax, jaxlib, and the NN libraries I'm using are shifting week to week.

This code base currently supports:
 * Flax Linen (https://github.com/google/flax/tree/master/flax/linen) -- for models, validation w/ pretrained weights, and training from scratch
 * Objax (https://github.com/google/objax) -- for model and model validation with pretrained weights

This is essentially an adaptation of my PyTorch EfficienNet generator code (https://github.com/rwightman/gen-efficientnet-pytorch and also found in https://github.com/rwightman/pytorch-image-models) to JAX.

I started this to
* learn JAX by working with familiar code / models as a starting point,
* figure out which JAX modelling interface libraries ('frameworks') I liked,
* compare the training / inference runtime traits of non-trivial models across combinations of PyTorch, JAX, GPU and TPU in order to drive cost optimizations for scaling up of future projects

Where are we at:
* Training works on single node, multi-GPU and TPU v3-8 for Flax Linen variants w/ Tensorflow Datasets based pipeline
* The Objax and Flax Linen (nn.compact) variants of models are working (for inference) 
* Weights are ported from PyTorch (my timm training) and Tensorflow (original paper author releases) and are organized in zoo of sorts (borrowed PyTorch code) 
* Tensorflow and PyTorch data pipeline based validation scripts work with models and weights. For PT pipeline with PT models and TF pipeline with TF models the results are pretty much exact.

TODO:
- [x] Fix model weight inits (working for Flax Linen variants)
- [x] Fix dropout/drop path impl and other training specifics (verified for Flax Linen variants)
- [ ] Add more instructions / help in the README on how to get an optimal environment with JAX up and running (with GPU support)
- [x] Add basic training code. The main point of this is to scale up training.
- [ ] Add more advance data augmentation pipeline 
- [ ] Training on lots of GPUs
- [ ] Training on lots of TPUs

Some odd things:
* Objax layers are reimplemented to make my initial work easier, scratch some itches, make more consistent with PyTorch (because why not?)
* Flax Linen layers are by default fairly consistent with Tensorflow (left as is)
* I use wrappers around Flax Linen layers for some argument consistency and reduced visual noise (no redundant tuples)
* I made a 'LIKE' padding mode, sort of like 'SAME' but different, hence the name. It calculates symmetric padding for PyTorch models.
* Models with Tensorflow 'SAME' padding and TF origin weights are prefixed with `tf_`. Models with PyTorch trained weights and symmetric PyTorch style padding ('LIKE' here) are prefixed with `pt_`
* I use `pt` and `tf` to refer to PyTorch and Tensorflow for both the models and environments. These two do not need to be used together. `pt` models with 'LIKE' padding will work fine running in a Tensorflow based environment and vice versa. I did this to show the full flexibility here, that one can use JAX models with PyTorch data pipelines and datasets or with Tensorflow based data pipelines and TFDS. 

## Models

Supported models and their paper's
* EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
* EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665
* EfficientNet (B0-B7) - https://arxiv.org/abs/1905.11946
* EfficientNet-EdgeTPU (S, M, L) - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
* MixNet - https://arxiv.org/abs/1907.09595
* MobileNet-V3 - https://arxiv.org/abs/1905.02244
* MobileNet-V2 - https://arxiv.org/abs/1801.04381
* MNASNet B1, A1 (Squeeze-Excite), and Small - https://arxiv.org/abs/1807.11626
* Single-Path NAS - https://arxiv.org/abs/1904.02877
* FBNet-C - https://arxiv.org/abs/1812.03443

Models by their config name w/ valid pretrained weights that should be working here:
```
pt_mnasnet_100
pt_semnasnet_100
pt_mobilenetv2_100
pt_mobilenetv2_110d
pt_mobilenetv2_120d
pt_mobilenetv2_140
pt_fbnetc_100
pt_spnasnet_100
pt_efficientnet_b0
pt_efficientnet_b1
pt_efficientnet_b2
pt_efficientnet_b3
tf_efficientnet_b0
tf_efficientnet_b1
tf_efficientnet_b2
tf_efficientnet_b3
tf_efficientnet_b4
tf_efficientnet_b5
tf_efficientnet_b6
tf_efficientnet_b7
tf_efficientnet_b8
tf_efficientnet_b0_ap
tf_efficientnet_b1_ap
tf_efficientnet_b2_ap
tf_efficientnet_b3_ap
tf_efficientnet_b4_ap
tf_efficientnet_b5_ap
tf_efficientnet_b6_ap
tf_efficientnet_b7_ap
tf_efficientnet_b8_ap
tf_efficientnet_b0_ns
tf_efficientnet_b1_ns
tf_efficientnet_b2_ns
tf_efficientnet_b3_ns
tf_efficientnet_b4_ns
tf_efficientnet_b5_ns
tf_efficientnet_b6_ns
tf_efficientnet_b7_ns
tf_efficientnet_l2_ns_475
tf_efficientnet_l2_ns
pt_efficientnet_es
pt_efficientnet_em
tf_efficientnet_es
tf_efficientnet_em
tf_efficientnet_el
pt_efficientnet_lite0
tf_efficientnet_lite0
tf_efficientnet_lite1
tf_efficientnet_lite2
tf_efficientnet_lite3
tf_efficientnet_lite4
pt_mixnet_s
pt_mixnet_m
pt_mixnet_l
pt_mixnet_xl
tf_mixnet_s
tf_mixnet_m
tf_mixnet_l
pt_mobilenetv3_large_100
tf_mobilenetv3_large_075
tf_mobilenetv3_large_100
tf_mobilenetv3_large_minimal_100
tf_mobilenetv3_small_075
tf_mobilenetv3_small_100
tf_mobilenetv3_small_minimal_100
```

## Environment

Working with JAX I've found the best approach for having a working GPU compatible environment that performs well is to use Docker containers based on the latest NVIDIA NGC releases. I've found it challenging or flaky getting local conda/pip venvs or Tensorflow docker containers working well with good GPU performance, proper NCCL distributed support, etc. I use CPU JAX install in conda env for dev/debugging.

### Dockerfiles

There are several container definitions in `docker/`. They use NGC containers as their parent image so you'll need to be setup to pull NGC containers: https://www.nvidia.com/en-us/gpu-cloud/containers/ . I'm currently using recent NGC containers w/ CUDA 11.1 support, the host system will need a very recent NVIDIA driver to support this but doesn't need a matching CUDA 11.1 / cuDNN 8 install.

Current dockerfiles:
* `pt_git.Dockerfile` - PyTorch 20.12 NGC as parent, CUDA 11.1, cuDNN 8. git (source install) of jaxlib, jax, objax, and flax.
* `pt_pip.Dockerfile` - PyTorch 20.12 NGC as parent, CUDA 11.1, cuDNN 8. pip (latest ver) install of jaxlib, jax, objax, and flax.
* `tf_git.Dockerfile` - Tensorflow 2 21.02 NGC as parent, CUDA 11.2, cuDNN 8. git (source install) of jaxlib, jax, objax, and flax.
* `tf_pip.Dockerfile` - Tensorflow 2 21.02 NGC as parent, CUDA 11.2, cuDNN 8. pip (latest ver) install of jaxlib, jax, objax, and flax.

The 'git' containers take some time to build jaxlib, they pull the masters of all respective repos so are up to the bleeding edge but more likely to have possible regression or incompatibilities that go with that. The pip install containers are quite a bit quicker to get up and running, based on the latest pip versions of all repos.

### Docker Usage (GPU)

1. Make sure you have a recent version of docker and the NVIDIA Container Toolkit setup (https://github.com/NVIDIA/nvidia-docker) 
2. Build the container `docker build -f docker/tf_pip.Dockerfile -t jax_tf_pip .`
3. Run the container, ideally map jeffnet and datasets (ImageNet) into the container
    * For tf containers, `docker run --gpus all -it -v /path/to/tfds/root:/data/ -v /path/to/efficientnet-jax/:/workspace/jeffnet --rm --ipc=host jax_tf_pip`
    * For pt containers, `docker run --gpus all -it -v /path/to/imagenet/root:/data/ -v /path/to/efficientnet-jax/:/workspace/jeffnet --rm --ipc=host jax_pt_pip`
4. Model validation w/ pretrained weights (once inside running container):
    * For tf, in `worskpace/jeffnet`, `python tf_linen_validate.py /data/ --model tf_efficientnet_b0_ns`
    * For pt, in `worskpace/jeffnet`, `python pt_objax_validate.py /data/validation --model pt_efficientnet_b0`
5. Training (within container)
    * In `worskpace/jeffnet`, `tf_linen_train.py --config train_configs/tf_efficientnet_b0-gpu_24gb_x2.py --config.data_dir /data`

### TPU

I've successfully used this codebase on TPU VM environments as is. Any of the `tpu_x8` training configs should work out of the box on a v3-8 TPU. I have not tackled training with TPU Pods.
