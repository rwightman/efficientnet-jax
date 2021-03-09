""" EfficientNet, MobileNetV3, etc Builder

Assembles EfficieNet and related network feature blocks from string definitions.
Handles stride, dilation calculations, and selects feature extraction points.

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging

from .block_utils import round_features


__all__ = ['EfficientNetBuilder']

_logger = logging.getLogger(__name__)


def _log_info_if(msg, condition):
    if condition:
        _logger.info(msg)


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, in_chs, block_defs, block_factory,
                 feat_multiplier=1.0, feat_divisor=8, feat_min=None,
                 output_stride=32, pad_type='', conv_layer=None, norm_layer=None, se_layer=None,
                 act_fn=None, drop_path_rate=0., feature_location='', verbose=False):
        assert output_stride in (32, 16, 8, 4, 2)
        self.in_chs = in_chs  # num input ch from stem
        self.block_defs = block_defs  # block types, arguments w/ structure
        self.block_factory = block_factory  # factory to build framework specific blocks
        self.feat_multiplier = feat_multiplier
        self.feat_divisor = feat_divisor
        self.feat_min = feat_min
        self.output_stride = output_stride
        self.act_fn = act_fn
        self.drop_path_rate = drop_path_rate
        self.default_args = dict(
            pad_type=pad_type,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
        )
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = verbose

        self.features = []  # information about feature maps, constructed during build

    def _round_channels(self, chs):
        return round_features(chs, self.feat_multiplier, self.feat_divisor, self.feat_min)

    def _make_block(self, block_type, block_args, stage_idx, block_idx, flat_idx, block_count):
        drop_path_rate = self.drop_path_rate * flat_idx / block_count
        # NOTE: block act fn overrides the model default
        act_fn = block_args['act_fn'] if block_args['act_fn'] is not None else self.act_fn
        act_fn = self.block_factory.get_act_fn(act_fn)  # map string acts to functions
        ba_overlay = dict(
            in_chs=self.in_chs, out_chs=self._round_channels(block_args['out_chs']), act_fn=act_fn,
            drop_path_rate=drop_path_rate, **self.default_args)
        block_args.update(ba_overlay)
        if 'fake_in_chs' in block_args and block_args['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            block_args['fake_in_chs'] = self._round_channels(block_args['fake_in_chs'])
        assert block_args['act_fn'] is not None

        _log_info_if(f'  {block_type.upper()} {block_idx}, Args: {str(block_args)}', self.verbose)
        if block_type == 'ir':
            block = self.block_factory.InvertedResidual(stage_idx, block_idx, **block_args)
        elif block_type == 'ds' or block_type == 'dsa':
            block = self.block_factory.DepthwiseSeparable(stage_idx, block_idx, **block_args)
        elif block_type == 'er':
            block = self.block_factory.EdgeResidual(stage_idx, block_idx, **block_args)
        elif block_type == 'cn':
            block = self.block_factory.ConvBnAct(stage_idx, block_idx, **block_args)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % block_type
        self.in_chs = block_args['out_chs']  # update in_chs for arg of next block

        return block

    def __call__(self):
        """ Build the blocks
        Return:
             List of stages (each stage being a list of blocks)
        """
        _log_info_if('Building model trunk with %d stages...' % len(self.block_defs), self.verbose)
        num_blocks = sum([len(x) for x in self.block_defs])
        flat_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        # if self.block_args[0][0]['stride'] > 1:
        #     # if the first block starts with a stride, we need to extract first level feat from stem
        #     self.features.append(dict(
        #         module='act1', num_chs=self.in_chs, stage=0, reduction=current_stride,
        #         hook_type='forward' if self.feature_location != 'bottleneck' else ''))

        # outer list of block_args defines the stacks
        for stage_idx, stage_defs in enumerate(self.block_defs):
            _log_info_if('Stack: {}'.format(stage_idx), self.verbose)

            blocks = []
            # each stage contains a list of block types and arguments
            for block_idx, block_def in enumerate(stage_defs):
                _log_info_if(' Block: {}'.format(block_idx), self.verbose)
                last_block = block_idx + 1 == len(stage_defs)
                block_type, block_args = block_def
                block_args = dict(**block_args)

                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:   # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                # extract_features = False
                # if last_block:
                #     next_stage_idx = stage_idx + 1
                #     extract_features = next_stage_idx >= len(self.block_defs) or \
                #         self.block_defs[next_stage_idx][0]['stride'] > 1

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        _log_info_if('  Converting stride to dilation to maintain output_stride=={}'.format(
                            self.output_stride), self.verbose)
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                blocks.append(self._make_block(block_type, block_args, stage_idx, block_idx, flat_idx, num_blocks))

                # stash feature module name and channel info for model feature extraction
                # if extract_features:
                #     feature_info = dict(stage=stage_idx + 1, reduction=current_stride)
                #     module_name = f'blocks.{stage_idx}.{block_idx}'
                #     leaf_name = feature_info.get('module', '')
                #     feature_info['module'] = '.'.join([module_name, leaf_name]) if leaf_name else module_name
                #     self.features.append(feature_info)

                flat_idx += 1  # incr flattened block idx (across all stages)
            stages.append(blocks)
        return stages
