from __future__ import division
import math 
import numpy as np
import tensorflow as tf

def int_shape(x):
    return list(map(int, x.get_shape()))
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

def discretized_mix_logistic_loss(y_hat, y, num_classes = 256, log_scale_min = -7.0, reduce = True):
    """
        Discretized mixture of logistic distributions loss
        Note: Assumed that the input is scaled to [-1, 1].

        Args:
            y_hat (Tensor): Predicted output (B x T x C)[Batch_size, time_length, channels]
            y (Tensor): Target (B x T x 1).
            num_classes (int): Number of classes
            log_scale_min (float):  the log scale minimum value
            reduce (bool): If True, the losses are averaged or summed for each minibatch.
    
        Returns:
            Tensor: loss
    """
   

    # Shapes
    y_shape = int_shape(y)
    y_hat_shape = int_shape(y_hat) 

     nr_mix = y_hat_shape[2] // 3
    
    # unpaccking parameters of mixture distribution
    # (B, T, nr_mix) x 3
    logit_probs = y_hat[:,:,:nr_mix]
    means = y_hat[:, :, nr_mix: 2 * nr_mix]
    log_scales = tf.maximum(y_hat[:, :, 2 * nr_mix: 3 * nr_mix], log_scale_min)

    # B x T x 1 --> B x T x num_mixtures
    y = tf.tile(y, [1, 1, nr_mix])
    # y = tf.squeeze(y, axis=2)  # [B, T, 1] --> [B, T]
    # y = tf.expand_dims(y, axis=-1) + tf.zeros(y_shape + [nr_mix])  # [B, T, nr_mix]

    # comulative distribution function(cdf) = 1 / (1 +  e ** -x) = tf.nn.sigmoid(x)
    centered_y = y - means
    inv_stdv = tf.exp(-log_scales)
    
    plus_in = inv_stdv *  (centered_y + 1. / (num_classes - 1))
    cdf_plus = tf.nn.sigmoid(plus_in)

    min_in = inv_stdv * (centered_Y - 1. / (num_classes - 1))
    cdf_min = tf.nn.sigmoid(min_in)

    # log probablity for edge case of 0 (before scaling)
    # tf.log(tf.nn.sigmoid(plus_in))
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)

    # log probablity for edge case of 255 (before scaling)
    # log(1 -  tf.nn.sigmoid(min_in))
    log_one_minus_cdf_min = - tf.nn.softplus(min_in)

    # probablity for other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # Log probablity in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_min = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(y < -0.999 , log_cdf_plus,
                         tf.where(x > 0.999 , log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5, 
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log((num_classes - 1)  / 2))))

    log_probs = log_probs + log_prob_from_logits(logit_probs)  # [B, T, nr_mix]
    final_log = log_sum_exp(log_probs)  # [B, T]

    return -tf.reduce_mean(final_log)  # negative log-likelihood
def sample_from_discretized_mix_logistic(y, log_scale_min = -7.0):
    """
        Sample from discretized mixture of logistic distributions

        Args:
            y(Tensor): B x T x C
            log_scale (float): log scale minimum value
    """
    y_shape = int_shape(y)
   
    nr_mix = y_hat_shape[2] // 3
    logit_probs = y[:, :, :nr_mix]  # [B, T, nr_mix]
    # sample mixture indicator from softmax
    sel = tf.one_hot(  # [B, T, nr_mix]
        tf.argmax(  # [B, T]
            logit_probs - tf.log(-tf.log(  # [B, T, nr_mix]
                tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))),
            2),
        depth=nr_mix, dtype=tf.float32)

    # select logistic parameters
    means = tf.reduce_sum(y[:, :, nr_mix: 2 * nr_mix] * sel, 2)

    log_scales = tf.maximum(tf.reduce_sum(
        y[:, :, 2 * nr_mix: 3 * nr_mix] * sel, 2), log_scale_min)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))  # inverse of sigmoid, [B, T]
    # x = tf.clip_by_value(x, -0.9999999, 0.9999999)  # ITU-Ts it necessary?
    x = = tf.minimum(tf.maximum(x, -1.), 1.)

    # negative log-likelihood
    z = (x - means) * tf.exp(-log_scales)  # z = (x - u) / S
    log_likelihood = z - log_scales - 2. * tf.nn.softplus(z)
    return x

