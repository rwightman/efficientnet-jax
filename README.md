# EfficientNet JAX - Flax Linen and Objax

This is very much a giant steaming work in progress. The two JAX dressings I am using -- Objax, and Flax Linen -- are also shifting week to week. No effort will be made to maintain backwards compatibility with thece here. It will break.

This is essentially an adaptation of my PyTorch EfficienNet generator code (https://github.com/rwightman/gen-efficientnet-pytorch and also found in https://github.com/rwightman/pytorch-image-models) to JAX.

I started this to:
* learn JAX by working with familiar code / models as a starting point
* figure out which JAX modelling interface libraries ('frameworks') I liked
* compare the training / inference runtime traits of non-trivial models across combinations of PyTorch, JAX, GPU and TPU in order to drive cost optimizations for scaling up of future projects

Where are we at:
* The Objax and Flax Linen (nn.compact) variants of models are working (for inference) 
* Weights are ported from PyTorch (my timm training) and Tensorflow (original paper author releases) and are organized in zoo of sorts (borrowed PyTorch code) 
* Tensorflow and PyTorch data pipeline based validation scripts work with models and weights. For PT pipeline with PT models and TF pipeline with TF models the results are pretty much exact.

TODO:
- [ ] Fixup model weight inits (not currently correct), fix dropout/drop path impl and other training specifics.
- [ ] Add more instructions / help in the README on how to get an optimal environment with JAX up and running (with GPU support)
- [ ] Add basic training code. The main point of this is to scale up training.
- [ ] Add more advance data augmentation pipeline 
- [ ] Training on lots of GPUs
- [ ] Training on lots of TPUs

Some odd things:
* Objax layers are reimplemented to make my initial work easier, scratch some itches, make more consistent with PyTorch (because why not?)
* Flax Linen layers are by default fairly consistent with Tensorflow (left as is)
* I use wrappers around Flax Linen layers for some argument consistency and reduced visual noise (no redundant tuples)
* I made a 'LIKE' padding mode, sort of like 'SAME' but different, hence the name. It calculates symmetric padding for PyTorch models.
* Models with Tensorflow 'SAME' padding and TF origin weights are prefixed with `tf_`. Models with PyTorch trained weights and symmetric PyTorch style padding ('LIKE' here) are prefixed with `pt_`

These models are valid w/ weights that currently (should be) working here:
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
