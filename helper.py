import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers.merge import _Merge


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        batch_size = k.shape(inputs[0])[0]
        weights = k.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


# Just report the mean output of the model (useful for WGAN)
def output(y_pred):
    return k.mean(y_pred)


# dummy loss
def no_loss():
    return k.zeros(shape=(1,))


def wasserstein_loss(y_true, y_pred):
    return k.mean(y_true * y_pred)


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def precision(y_true):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    # return precision
    p = as_keras_metric(tf.metrics.precision)
    return p(y_true)


def recall(y_true):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # recall = true_positives / (possible_positives + K.epsilon())
    r = as_keras_metric(tf.metrics.recall)
    return r(y_true)


def f1_score(y_true):
    p = precision(y_true)
    r = recall(y_true)
    return (2 * p * r) / (p + r + k.epsilon())


def gradient_penalty_loss(y_pred, averaged_samples):
    def _compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    # gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = _compute_gradients(y_pred, [averaged_samples])[0]
    gradients_sqr = k.square(gradients)
    gradients_sqr_sum = k.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = k.sqrt(gradients_sqr_sum)
    gradient_penalty = k.square(1 - gradient_l2_norm)
    return k.mean(gradient_penalty)


# wrapper for using tensorflow metrics in keras
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(args, **kwargs)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


def plot(path, log):
    for key, vals in log.items():
        xs = list(range(len(vals)))
        ys = vals

        plt.clf()
        plt.plot(xs, ys)
        plt.xlabel('iteration')
        plt.ylabel(key)

        plt.savefig(os.path.join(path, key))