import tensorflow as tf
from neuralgym.ops.summary_ops import scalar_summary


def gan_sn_patch_gan_loss(pos, neg, name='gan_sn_patch_gan_loss'):
    
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    with tf.variable_scope(name):
        hinge_pos = tf.reduce_mean(tf.nn.relu(1-pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1+neg))
        scalar_summary('pos_hinge_avg', hinge_pos)
        scalar_summary('neg_hinge_avg', hinge_neg)
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
    return g_loss, d_loss


def kernel_spectral_norm(kernel, iteration=1, name='kernel_sn'):
    # spectral_norm
    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
        return input_x_norm
    with tf.variable_scope(name) as scope:
        w_shape = kernel.get_shape().as_list()
        w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
        u = tf.get_variable(
            'u', shape=[1, w_shape[-1]],
            initializer=tf.truncated_normal_initializer(),
            trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1

        u_hat, v_hat,_ = power_iteration(u, iteration)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


class Conv2DSepctralNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        super(Conv2DSepctralNorm, self).build(input_shape)
        self.kernel = kernel_spectral_norm(self.kernel)


def conv2d_spectral_norm(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = Conv2DSepctralNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)