from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def log_sum_exp(x):
    """
        Numerically stable log_sum_exp implementation that prevents overflow
        log_sum_exp: log(sum(exp(x), axis=-1))
        
    """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims = True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2),axis))

def log_prob_from_logits(x):
    """
        Numerically stable log_softmax implementation that prevents overflow
    """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims = True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x -m), axis, keep_dims = True))

def discretized_mix_logistic_loss(x, l, nr_mix, quantization_channels=65536, sum_all=True, scale_target=1):
    """ Negative? log-likelihood for mixture of discretized logistics,
    assumes the data has been rescaled to [-1,1] interval
    Args:
        x: [B, T, 1], value interal: [-1, 1]
        l: [B, T, nr_mix*3]
        nr_mix: number of logistic mixtures
    """
    x = tf.squeeze(x, axis=2)  # [B, T, 1] --> [B, T]
    # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :nr_mix]  # [B, T, nr_mix]
    means = l[:, :, nr_mix: 2 * nr_mix]  # [B, T, nr_mix]
    log_scales = tf.maximum(l[:, :, 2 * nr_mix: 3 * nr_mix],
                            LOG_SCALE_MINIMUM)  # [B, T, nr_mix]  modified from -7 to -14 to fit audio data

    x = tf.expand_dims(x, axis=-1) + tf.zeros(x.get_shape().as_list() + [nr_mix])  # [B, T, nr_mix]
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)  # [B, T, nr_mix]

    delta = (scale_target * 1.) / quantization_channels
    plus_in = inv_stdv * (centered_x + delta)  # [B, T, nr_mix]
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - delta)  # [B, T, nr_mix]
    cdf_min = tf.nn.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases

    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    
    log_probs = tf.where(x < -0.999 * scale_target, log_cdf_plus,  # [B, T, nr_mix]
                         tf.where(x > 0.999 * scale_target, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(32767.5 / scale_target))))

    log_probs = log_probs + log_prob_from_logits(logit_probs)  # [B, T, nr_mix]
    final_log = log_sum_exp(log_probs)  # [B, T]
    return -tf.reduce_mean(final_log)  # negative log-likelihood
def sample_from_discretized_mix_logistic(l, nr_mix, scale_target=1):
    """ Sample from discretized logistic mixtures
    l: [B, T, nr_mix*3]
    nr_mix: number of mixtures
    """
    logit_probs = l[:, :, :nr_mix]  # [B, T, nr_mix]
    # sample mixture indicator from softmax
    sel = tf.one_hot(  # [B, T, nr_mix]
        tf.argmax(  # [B, T]
            logit_probs - tf.log(-tf.log(  # [B, T, nr_mix]
                tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))),
            2),
        depth=nr_mix, dtype=tf.float32)

    # select logistic parameters
    means = tf.reduce_sum(l[:, :, nr_mix: 2 * nr_mix] * sel, 2)  # [B, T]

    log_scales = tf.maximum(tf.reduce_sum(  # [B, T]
        l[:, :, 2 * nr_mix: 3 * nr_mix] * sel, 2), LOG_SCALE_MINIMUM)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)  # [B, T]
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))  # inverse of sigmoid, [B, T]
    x = x / scale_target
    x = tf.clip_by_value(x, -0.9999999, 0.9999999)  # ITU-Ts it necessary?

    # negative log-likelihood
    z = (x - means) * tf.exp(-log_scales)  # z = (x - u) / S
    log_likelihood = z - log_scales - 2. * tf.nn.softplus(z)
    return x, log_likelihood
