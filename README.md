# EfficientNet JAX - Flax Linen and Objax

This is very much a giant steaming work in progress. The two JAX dressings I am using -- Objax, and Flax Linen -- are also shifting week to week. No effort will be made to maintain backwards compatibility with thece here. It will break.

This is essentially an adaptation of my PyTorch EfficienNet generator code (https://github.com/rwightman/gen-efficientnet-pytorch and also found in https://github.com/rwightman/pytorch-image-models) to JAX.

I started this to:
* learn JAX by working with familiar code / models as a starting point
* figure out which JAX modelling interface libraries ('frameworks') I liked
* compare the training / inference runtime traits of non-trivial models across combinations of PyTorch, JAX, GPU and TPU in order to drive cost optimizations for scaling up of future projects

Where are we at:
* The Objax and Flax Linen (nn.Compact) variants of models are working (for inference) 
* Weights are ported from PyTorch (my timm training) and Tensorflow (original paper author releases) and are organized in zoo of sorts (borrowed PyTorch code) 
* Tensorflow and PyTorch data pipeline based validation scripts work with models and weights. For PT pipeline with PT models and TF pipeline with TF models the results are pretty much exact.

TODO:
-[ ] Fixup model weight inits (not currently correct), fix dropout/drop path impl and other training specifics.
-[ ] Add more instructions / help in the README on how to get an optimal environment with JAX up and running (with GPU support)
-[ ] Add basic training code. The main point of this is to scale up training.
-[ ] Add more advance data augmentation pipeline 
-[ ] Training on lots of GPUs
-[ ] Training on lots of TPUs

Some odd things:
* Objax layers are reimplemented to make my initial work easier, scratch some itches, make more consistent with PyTorch (because why not?)
* Flax Linen layers are by default fairly consistent with Tensorflow (left as is)
* I use wrappers around Flax Linen layers for some argument consistency and reduced visual noise (no redundant tuples)
* I made a 'LIKE' padding mode, sort of like 'SAME' but different, hence the name. It calculates symmetric padding for PyTorch models.
