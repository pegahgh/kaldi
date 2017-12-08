# Copyright 2017 Pegah Ghahremani
# Apache 2.0.

""" This module contains layer types for processig raw waveform frames.
"""

from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is used for frequency-domain filter learning.
# This class is for parsing lines like
# 'preprocess-fft-abs-lognorm-affine-log-layer fft-dim=512 num-left-inputs=1'
# 'num-right-inputs=2 l2-reg=0.001'
# preprocess : applies windowing and pre-emphasis on input frames.
# fft : compute real and imaginary part of discrete cosine transform
#       using sine and cosine transform.
# abs : computes absolute value of real and complex part of fft.
# lognorm : normalize input in log-space using batchnorm followed by per-element
#           scale and offset.
# affine : filterbank learned using AffineComponent

class XconfigFftFilterLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in ['preprocess-fft-abs-lognorm-affine-log-layer',
                               'preprocess-fft-abs-norm-affine-log-layer',
                               'preprocess-fft-abs-log-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = { 'input':'[-1]',
                        'dim': -1,
                        'max-change' : 0.75,
                        'target-rms' : 1.0,
                        'learning-rate-factor' : 1.0,
                        'max-change' : 0.75,
                        'max-param-value' : 1.0,
                        'min-param-value' : 0.0,
                        'l2-regularize' : 0.005,
                        'learning-rate-factor' : 1,
                        'dim' : -1,
                        'write-init-config' : True,
                        'num-filters' : 100,
                        'sin-transform-file' : '',
                        'cos-transform-file' : '',
                        'half-fft-range' : False} # l2-regularize and min-param-value
                                                   # and max-param-value affects
                                                   # layers affine layer.
    def check_configs(self):
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        if self.config['max-param-value'] < self.config['min-param-value']:
            raise RuntimeError("max-param-value {0} should be larger than "
                               "min-param-value {1}."
                               "".format(self.config['max-param-value'],
                                         self.config['min-param-value']))

        if self.config['sin-transform-file'] is None:
            raise RuntimeError("sin-transform-file must be set.")

        if self.config['cos-transform-file'] is None:
            raise RuntimeError("cos-transform-file must be set.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        return '{0}.{1}'.format(self.name, last_nonlinearity)


    def output_dim(self):
        split_layer_name = self.layer_type.split('-')
        if 'affine' in split_layer_name:
            return self.config['num-filters']
        else:
            input_dim = self.descriptors['input']['dim']
            fft_dim = (2**(input_dim-1).bit_length())
            half_fft_range = self.config['half-fft-range']
            output_dim = (fft_dim/2 if half_fft_range is True else fft_dim)
            return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            if len(line) == 2:
                # 'ref' or 'final' tuple already exist in the line
                # These lines correspond to fft component.
                # which contains FixedAffineComponent.
                assert(line[0] == 'init' or line[0] == 'ref' or line[0] == 'final')
                ans.append(line)
            else:
                for config_name in ['ref', 'final']:
                    ans.append((config_name, line))
        return ans

    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, nonlinearities):
        dim = self.config['dim']
        min_param_value = self.config['min-param-value']
        max_param_value = self.config['max-param-value']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        #ng_affine_options = self.config['ng-affine-options']
        learning_rate_factor= self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        cos_file = self.config['cos-transform-file']
        sin_file = self.config['sin-transform-file']
        num_filters = self.config['num-filters']
        l2_regularize = self.config['l2-regularize']
        half_fft_range = self.config['half-fft-range']
        fft_dim = (2**(input_dim-1).bit_length())
        cur_dim = input_dim
        cur_node = input_desc
        configs = []
        for nonlinearity in nonlinearities:
            if nonlinearity == 'preprocess':
                configs.append('component name={0}.preprocess type=ShiftInputComponent '
                               'input-dim={1} output-dim={1} dither=0.0 max-shift=0.0 '
                               'preprocess=true'.format(self.name, cur_dim))

                configs.append('component-node name={0}.preprocess '
                               'component={0}.preprocess input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.preprocess'.format(self.name)

            elif nonlinearity == 'fft':
                #if self.config['write-init-config']:
                #    line = ('output-node name=output input={0}'
                #            ''.format(input_desc))
                #    configs.append(('init', line))
                output_dim = (fft_dim/2 if half_fft_range is True else fft_dim)
                line = ('component name={0}.cosine type=FixedAffineComponent '
                       'matrix={1}'
                       ''.format(self.name, cos_file))
                configs.append(('final', line))

                line = ('component name={0}.cosine type=FixedAffineComponent '
                        'input-dim={1} output-dim={2}'
                        ''.format(self.name, cur_dim, output_dim))
                configs.append(('ref', line))

                line = ('component-node name={0}.cosine component={0}.cosine '
                        'input={1}'.format(self.name, cur_node))
                configs.append(('final', line))
                configs.append(('ref', line))

                line = ('component name={0}.sine type=FixedAffineComponent '
                        'matrix={1}'.format(self.name, sin_file))
                configs.append(('final', line))

                line = ('component name={0}.sine type=FixedAffineComponent '
                        'input-dim={1} output-dim={2}'
                        ''.format(self.name, cur_dim, output_dim))
                configs.append(('ref', line))

                line = ('component-node name={0}.sine component={0}.sine '
                        'input={1}'.format(self.name, cur_node))
                configs.append(('final', line))
                configs.append(('ref', line))

                cur_node = []
                if half_fft_range:
                    cur_node.append('{0}.cosine'.format(self.name))
                    cur_node.append('{0}.sine'.format(self.name))
                else:
                    configs.append('dim-range-node name={0}.sine.half input-node={0}.sine '
                                   'dim-offset=0 dim={1}'.format(self.name, fft_dim/2))
                    configs.append('dim-range-node name={0}.cosine.half input-node={0}.cosine '
                                   'dim-offset=0 dim={1}'.format(self.name, fft_dim/2))
                    cur_node.append('{0}.cosine.half'.format(self.name))
                    cur_node.append('{0}.sine.half'.format(self.name))
                cur_dim = fft_dim / 2
            elif nonlinearity == 'abs2':
                assert(len(cur_node) == 2 and
                       cur_node[0] == '{0}.cosine'.format(self.name) and
                       cur_node[1] == '{0}.sine'.format(self.name))
                configs.append('component name={0}.cos.sqr type=ElementwiseProductComponent '
                               'input-dim={1} output-dim={2}'
                               ''.format(self.name, cur_dim * 2, cur_dim))
                configs.append('component-node name={0}.cos.sqr component={0}.cos.sqr '
                               'input=Append({1},{1})'
                               ''.format(self.name, cur_node[0]))

                configs.append('component name={0}.sin.sqr type=ElementwiseProductComponent '
                               'input-dim={1} output-dim={2}'
                               ''.format(self.name, cur_dim * 2, cur_dim))
                configs.append('component-node name={0}.sin.sqr component={0}.cos.sqr '
                               'input=Append({1},{1})'
                               ''.format(self.name, cur_node[1]))
                configs.append('component name={0}.abs type=NoOpComponent dim={1}'
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.abs component={0}.abs '
                               'input=Sum({0}.sin.sqr, {0}.cos.sqr)'
                               ''.format(self.name))
                cur_node = '{0}.abs'.format(self.name)

            elif nonlinearity == 'abs':
                assert(len(cur_node) == 2 and
                       cur_node[0] == '{0}.cosine'.format(self.name) and
                       cur_node[1] == '{0}.sine'.format(self.name))
                permute_vec = []
                for i in range(fft_dim/2):
                    permute_vec.append(i)
                    permute_vec.append(i+fft_dim/2)
                permute_vec_str = ','.join([str(x) for x in permute_vec])
                configs.append('component name={0}.permute type=PermuteComponent '
                               'column-map={1}'.format(self.name, permute_vec_str))
                configs.append('component-node name={0}.permute component={0}.permute '
                               'input=Append({1},{2})'
                               ''.format(self.name, cur_node[0], cur_node[1]))

                configs.append('component name={0}.abs type=PnormComponent '
                               'input-dim={1} output-dim={2} p=2.0'
                               ''.format(self.name, fft_dim, fft_dim/2))
                configs.append('component-node name={0}.abs component={0}.abs '
                               'input={0}.permute'.format(self.name))
                cur_node = '{0}.abs'.format(self.name)
                cur_dim = fft_dim / 2

            elif nonlinearity == 'lognorm':
                assert(isinstance(cur_node, str))
                configs.append('component name={0}.norm.log type=LogComponent '
                               'dim={1} log-floor=1e-4 additive-offset=false '
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm.log component={0}.norm.log '
                               'input={1}'.format(self.name, cur_node))
                configs.append('component name={0}.norm.batch type=BatchNormComponent '
                               'dim={1} target-rms={2} '
                               ''.format(self.name, cur_dim, target_rms))
                configs.append('component-node name={0}.norm.batch '
                               'component={0}.norm.batch '
                               'input={0}.norm.log'.format(self.name))
                configs.append('component name={0}.norm.so type=ScaleAndOffsetComponent '
                               'dim={1} max-change=0.5 '
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm.so component={0}.norm.so '
                               'input={0}.norm.batch '.format(self.name))
                configs.append('component name={0}.norm.exp type=ExpComponent dim={1} '
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm.exp component={0}.norm.exp '
                               'input={0}.norm.so'.format(self.name))
                cur_node = '{0}.norm.exp'.format(self.name)
                cur_dim = fft_dim / 2


            elif nonlinearity == 'lognorm2':
                configs.append("component name={0}.lognorm type=CompositeComponent "
                               "num-components=4 "
                               "component1='type=LogComponent dim={1} log-floor=1e-4 additive-offset=false' "
                               "component2='type=BatchNormComponent dim={1} target-rms={2}' "
                               "component3='type=ScaleAndOffsetComponent dim={1} max-change=0.5' "
                               "component4='type=ExpComponent dim={1}' "
                               "".format(self.name, cur_dim, target_rms))
                configs.append('component-node name={0}.lognorm '
                               'component={0}.lognorm input={1}'
                               ''.format(self.name, cur_node))

                cur_node = '{0}.lognorm'.format(self.name)
                cur_dim = fft_dim / 2

            elif nonlinearity == 'affine':
                configs.append('component name={0}.filterbank type=AffineComponent '
                               'input-dim={1} output-dim={2} max-change={3} '
                               'min-param-value={4} max-param-value={5} '
                               'bias-stddev=0.0 l2-regularize={6}'
                               ''.format(self.name, cur_dim, num_filters, max_change,
                                         min_param_value, max_param_value,
                                         l2_regularize))
                configs.append('component-node name={0}.filterbank '
                               'component={0}.filterbank input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.filterbank'.format(self.name)
                cur_dim = num_filters
            elif nonlinearity == 'log':
                configs.append('component name={0}.log type=LogComponent '
                               'log-floor=1e-4 additive-offset=false dim={1}'
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.log '
                               'component={0}.log input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.log'.format(self.name)
                cur_dim = fft_dim / 2

            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   "".format(nonlinearity))
        return configs
