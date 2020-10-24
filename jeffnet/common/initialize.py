
#
# def _init_weight_goog(m, n='', fix_group_fanout=True):
#     """ Weight initialization as per Tensorflow official implementations.
#
#     Args:
#         m (nn.Module): module to init
#         n (str): module name
#         fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs
#
#     Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
#     * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
#     * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
#     """
#     if isinstance(m, nn.Conv2d):
#         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         if fix_group_fanout:
#             fan_out //= m.groups
#         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1.0)
#         m.bias.data.zero_()
#     elif isinstance(m, nn.Linear):
#         fan_out = m.weight.size(0)  # fan-out
#         fan_in = 0
#         if 'routing_fn' in n:
#             fan_in = m.weight.size(1)
#         init_range = 1.0 / math.sqrt(fan_in + fan_out)
#         m.weight.data.uniform_(-init_range, init_range)
#         m.bias.data.zero_()
#
#
# def efficientnet_init_weights(model: nn.Module, init_fn=None):
#     init_fn = init_fn or _init_weight_goog
#     for n, m in model.named_modules():
#         init_fn(m, n)
