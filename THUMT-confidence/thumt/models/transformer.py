# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x, epsilon=1e-12)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, attention_dropout=0., residual_dropout=0.,relu_dropout=0., dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, attention_dropout=0., residual_dropout=0., relu_dropout=0., state=None,
                        dtype=None, scope=None, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias],reuse=tf.AUTO_REUSE):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        #params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    #if params.position_info_type == 'absolute':
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, params.attention_dropout, params.residual_dropout, params.relu_dropout, scope="encoder")


    return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    #if params.position_info_type == 'absolute':
    decoder_input = layers.attention.add_timing_signal(decoder_input)
    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params, params.attention_dropout, params.residual_dropout, params.relu_dropout, scope='decoder')
        
    else:
        # When inference decoder_input is one time step
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, 0., 0., 0., state=state["decoder"],
                                              scope='decoder')

        decoder_output, decoder_state = decoder_outputs
        # [batch_size, 1, vocab_size] => [batch_size, vocab_size]
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # with label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)
    ce = tf.reshape(ce, tf.shape(tgt_seq))
    '''
    if mode == "eval":
        #return ce * tgt_mask
        return -tf.reduce_sum(ce * tgt_mask, axis=1)
    '''
    mle_loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    # without label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)
    ce = tf.reshape(ce, tf.shape(tgt_seq))

    # bridge layer
    # [batch_size*length, vocab_size]

    soft_input = tf.nn.softmax(logits)
    soft_input = tf.stop_gradient(soft_input, name='stop_grad_softinput')
    tf.get_variable_scope().reuse_variables()
    # [batch_size*length, hidden_size] 
    soft_input = tf.matmul(soft_input, tgt_embedding) 

    if params.bridge_input_scale == "sqrt_depth":
        soft_input = soft_input * (hidden_size ** 0.5)

    if params.use_bridge_dropout:
        keep_prob = 1.0 - params.residual_dropout
        soft_input = tf.nn.dropout(soft_input, keep_prob)

    # [batch_size, length, hidden_size]
    soft_input = tf.reshape(soft_input, tf.concat([tf.shape(tgt_seq), [hidden_size]],0)) 
    soft_input = soft_input * tf.expand_dims(tgt_mask, -1)
    soft_input = tf.pad(soft_input, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    soft_input = layers.attention.add_timing_signal(soft_input)
    

    # get rand_trg input, make sure happen_prob&replace_prob=1.0
    rand_targets = tf.gather(tgt_embedding, features['rand_target'])
    if params.multiply_embedding_mode == "sqrt_depth":
        rand_targets = rand_targets * (hidden_size ** 0.5)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        rand_targets = tf.nn.dropout(rand_targets, keep_prob)

    rand_targets = rand_targets * tf.expand_dims(tgt_mask, -1)
    rand_decoder_input = tf.pad(rand_targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    rand_decoder_input = layers.attention.add_timing_signal(rand_decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        rand_decoder_input = tf.nn.dropout(rand_decoder_input, keep_prob)

    # For each t step, select y_{t-1} from random trg when ce < -log(0.8) ce
    # from golden trg when ce > -log(0.3)
    # then from soft trg when -log(0.8) < ce < -log(0.3)

    # params.select_random_trg_prob = 0.8
    # params.select_golden_trg_prob = 0.3
    ce = tf.expand_dims(ce, -1) # [b, s, 1]
    ce = tf.tile(ce, [1, 1, hidden_size]) # [b, s, h]
    # left shift, pad 0 at right side, clip the first one
    #ce = tf.pad(ce, [[0, 0], [0, 1], [0, 0]])[:, 1:, :]

    new_input = tf.where(tf.less(ce, -tf.log(params.select_random_trg_prob)),
                              x=rand_decoder_input, y=soft_input, name='select_random_trg_or_soft_inpu_op')

    new_input = tf.where(tf.less(-tf.log(params.select_golden_trg_prob), ce),
                              x=decoder_input, y=new_input, name='select_golden_or_soft_input_op')


    bridge_output = transformer_decoder(new_input, encoder_output,
                                        dec_attn_bias, enc_attn_bias,
                                        params, params.attention_dropout, params.residual_dropout, params.relu_dropout, scope='decoder')
    bridge_output = tf.reshape(bridge_output, [-1, hidden_size])
    bridge_logits = tf.matmul(bridge_output, weights, False, True)
    
    bridge_ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=bridge_logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    bridge_ce = tf.reshape(bridge_ce, tf.shape(tgt_seq))

    if mode == "eval":
        return bridge_ce * tgt_mask
        #return -tf.reduce_sum(bridge_ce * tgt_mask, axis=1)

    bridge_loss = tf.reduce_sum(bridge_ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return (mle_loss, bridge_loss)


def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output
    }
    
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                losses = model_graph(features, "train", params) # (mle_loss, bridge_loss)
                return losses

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=1024,
            filter_size=4096,
            num_heads=16,
            num_encoder_layers=15,
            num_decoder_layers=6,
            attention_dropout=0.1,
            residual_dropout=0.1,
            relu_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="layer_norm",
            layer_postprocess="none",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="relative",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=8,
            bridge_threshold=1.0,
            bridge_stop_gradient=True,
            bridge_input_scale="",
            use_bridge_dropout=False,
            is_finetuning=True,
            select_golden_trg_prob=0.3,
            select_random_trg_prob=0.8,
            random_trg_from_vocab=False,
            mle_rate=1.0,
            disable_first_dropout=False
        )

        return params

