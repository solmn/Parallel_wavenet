"""Unit tests for the mixture model"""

import numpy as np
import tensorflow as tf
import librosa
from wavenet import  discretized_mix_logistic_loss, sample_from_discretized_mix_logistic,log_sum_exp,log_prob_from_logits
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
s1 = tf.Session()
s2 = tf.Session()
def test_log_softmax():
    x = tf.random_uniform((2,16000, 30), seed=1)
   
    y = log_prob_from_logits(x)
    y_hat = tf.nn.log_softmax(x, axis=-1)
    # checks elementwise equal with tolerance
   
    y = s1.run(y)
    y_hat = s2.run(y_hat)
    assert np.allclose(y_hat, y)
def test_log_sum_exp():
    x = tf.random_uniform((2, 16000, 30), seed=1)
    y = tf.reduce_logsumexp(x,-1)
    y_hat = log_sum_exp(x)
    y = s1.run(y)
    y_hat = s2.run(y_hat)
    assert np.allclose(y, y_hat)

def test_mixture_loss():
    x, sr = librosa.load(librosa.util.example_audio_file())
    assert sr == 22050

    y_hat = tf.random_uniform(dtype=tf.float32, shape=(1, len(x), 30))
    y = x.reshape(1, len(x), 1)
    loss = discretized_mix_logistic_loss(y_hat, y, reduce=False)
    logit_probs = y_hat[:, :, :10]
    temp = tf.random_uniform(tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5) 
    temp = logit_probs - tf.log(-tf.log(temp)) 
    argmax = tf.argmax(temp, -1) 
    print(s1.run(argmax))
    assert loss.shape == y_hat.shape



