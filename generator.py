from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten  
from tensorflow.keras.initializers import glorot_uniform as xavier_initializer 
from contextlib import contextmanager
import numpy as np


def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=input_layer.get_shape().as_list(),
                             mean=0.0,
                             stddev=std,
                             dtype=tf.float32)
    return input_layer + noise

def sample_random_walk(batch_size, dim):
    rw = np.zeros((batch_size, dim))
    rw[:, 0] = np.random.randn(batch_size)
    for b in range(batch_size):
        for di in range(1, dim):
            rw[b, di] = rw[b, di - 1] + np.random.randn(1)
    # normalize to m=0 std=1
    mean = np.mean(rw, axis=1).reshape((-1, 1))
    std = np.std(rw, axis=1).reshape((-1, 1))
    rw = (rw - mean) / std
    return rw

def scalar_summary(name, x):
    return tf.summary.scalar(name, x)

def histogram_summary(name, x):
    return tf.summary.histogram(name, x)

def tensor_summary(name, x):
    return tf.summary.tensor_summary(name, x)

def audio_summary(name, x, sampling_rate=16e3):
    return tf.summary.audio(name, x, sampling_rate)

def minmax_normalize(x, x_min, x_max, o_min=-1., o_max=1.):
    return (o_max - o_min) / (x_max - x_min) * (x - x_max) + o_max

def minmax_denormalize(x, x_min, x_max, o_min=-1., o_max=1.):
    return minmax_normalize(x, o_min, o_max, x_min, x_max)

def downconv(x, output_dim, kwidth=5, pool=2, init=None, uniform=False,
             bias_init=None, name='downconv'):
    """ Downsampled convolution 1d """
    x2d = tf.expand_dims(x, 2)
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable('W', [kwidth, 1, x.get_shape()[-1], output_dim],
                                      initializer=w_init)
        conv = tf.nn.conv2d(x2d, W, strides=[1, pool, 1, 1], padding='SAME')
        if bias_init is not None:
            b = tf.compat.v1.get_variable('b', [output_dim],
                                          initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())
        # reshape back to 1d
        conv = tf.reshape(conv, conv.get_shape().as_list()[:2] +
                          [conv.get_shape().as_list()[-1]])
        return conv

def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu, name='hw'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in range(layer_size):
        lin_scope = '{}_output_lin_{}'.format(name, idx)
        output = f(tf.compat.v1.nn.rnn_cell._linear(output, size, 0, scope=lin_scope))
        transform_scope = '{}_transform_lin_{}'.format(name, idx)
        transform_gate = tf.sigmoid(
            tf.compat.v1.nn.rnn_cell._linear(input_, size, 0, scope=transform_scope) + bias)
        carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

    return output

def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)

def prelu(x, name='prelu', ref=False):
    in_shape = x.get_shape().as_list()
    with tf.compat.v1.variable_scope(name):
        # make one alpha per feature
        alpha = tf.compat.v1.get_variable('alpha', in_shape[-1],
                                          initializer=tf.constant_initializer(0.),
                                          dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        if ref:
            # return ref to alpha vector
            return pos + neg, alpha
        else:
            return pos + neg

def conv1d(x, kwidth=5, num_kernels=1, init=None, uniform=False, bias_init=None,
           name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.compat.v1.variable_scope(name):
        # filter shape: [kwidth, in_channels, num_kernels]
        W = tf.compat.v1.get_variable('W', [kwidth, in_channels, num_kernels],
                                      initializer=w_init
                                      )
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.compat.v1.get_variable('b', [num_kernels],
                                          initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv

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
                          [tf.compat.v1.div(shape[0], dilation), -1, shape[2]])

def atrous_conv1d(value, dilation, kwidth=3, num_kernels=1,
                  name='atrous_conv1d', bias_init=None, stddev=0.02):
    input_shape = value.get_shape().as_list()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    with tf.compat.v1.variable_scope(name):
        weights_init = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
        # filter shape: [kwidth, in_channels, output_channels]
        filter_ = tf.compat.v1.get_variable('w', [kwidth, in_channels, num_kernels],
                                            initializer=weights_init,
                                            )
        padding = [[0, 0], [(kwidth//2) * dilation, (kwidth//2) * dilation],
                  [0, 0]]
        padded = tf.pad(value, padding, mode='SYMMETRIC')
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, input_shape[1], num_kernels])
        if bias_init is not None:
            b = tf.compat.v1.get_variable('b', [num_kernels],
                                          initializer=tf.constant_initializer(bias_init))
            result = tf.add(result, b)
        return result

def residual_block(input_, dilation, kwidth, num_kernels=1,
                   bias_init=None, stddev=0.02, do_skip=True,
                   name='residual_block'):
    print('input shape to residual block: ', input_.get_shape())
    with tf.compat.v1.variable_scope(name):
        h_a = atrous_conv1d(input_, dilation, kwidth, num_kernels,
                            bias_init=bias_init, stddev=stddev)
        h = tf.tanh(h_a)
        # apply gated activation
        z_a = atrous_conv1d(input_, dilation, kwidth, num_kernels,
                            name='conv_gate', bias_init=bias_init,
                            stddev=stddev)
        z = tf.nn.sigmoid(z_a)
        print('gate shape: ', z.get_shape())
        # element-wise apply the gate
        gated_h = tf.multiply(z, h)
        print('gated h shape: ', gated_h.get_shape())
        # make res connection
        h_ = conv1d(gated_h, kwidth=1, num_kernels=1,
                    init=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                    name='residual_conv1')
        res = h_ + input_
        print('residual result: ', res.get_shape())
        if do_skip:
            # make skip connection
            skip = conv1d(gated_h, kwidth=1, num_kernels=1,
                          init=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                          name='skip_conv1')
            return res, skip
        else:
            return res

def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.
    # Raises
        ValueError: In case `x.shape[axis]` is undefined.
    # Returns
        A tensor.
    """
    x_shape = x.get_shape().as_list()
    if x_shape[axis] is None:
        raise ValueError('Axis ' + str(axis) + ' of input tensor '
                         'should have a defined dimension, but is None. '
                         'Full tensor shape: ' + str(tuple(x_shape)) + '. '
                         'Typically you need to pass a fully-defined '
                         '`input_shape` argument to your first layer.')
    splits = tf.split(x, num_or_size_splits=x_shape[axis], axis=axis)
    x_rep = [s for s in splits for _ in range(rep)]
    return tf.concat(x_rep, axis)

def nn_deconv(x, kwidth=5, dilation=2, init=None, uniform=False,
              bias_init=None, name='nn_deconv1d'):
    interp_x = repeat_elements(x, dilation, 1)
    dec = conv1d(interp_x, kwidth=5, num_kernels=1, init=init, uniform=uniform, 
                 bias_init=bias_init, name=name, padding='SAME')
    return dec

def deconv(x, output_shape, kwidth=5, dilation=2, init=None, uniform=False,
           bias_init=None, name='deconv1d'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 3
    x2d = tf.expand_dims(x, 2)
    o2d = output_shape[:2] + [1] + [output_shape[-1]]
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable('W', [kwidth, 1, out_channels, in_channels],
                                      initializer=w_init
                                      )
        try:
            deconv = tf.nn.conv2d_transpose(x2d, W, output_shape=o2d,
                                            strides=[1, dilation, 1, 1])
        except AttributeError:
            deconv = tf.nn.deconv2d(x2d, W, output_shape=o2d,
                                    strides=[1, dilation, 1, 1])
        if bias_init is not None:
            b = tf.compat.v1.get_variable('b', [out_channels],
                                          initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        deconv = tf.reshape(deconv, output_shape)
        return deconv

def conv2d(input_, output_dim, k_h, k_w, stddev=0.05, name="conv2d", with_w=False):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                      initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
        if with_w:
            return conv, w
        else:
            return conv

@contextmanager
def variables_on_gpu0():
    old_fn = tf.compat.v1.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/gpu:0"):
            return old_fn(*args, **kwargs)
    tf.compat.v1.get_variable = new_fn
    yield
    tf.compat.v1.get_variable = old_fn

def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across towers.
    Note that this function provides a sync point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
        list is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
