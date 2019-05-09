import tensorflow as tf
from neuralgym.ops.summary_ops import scalar_summary


def gan_sn_patch_gan_loss(pos, neg, name='gan_sn_patch_gan_loss'):
    
    with tf.variable_scope(name):
        ones_x = tf.ones_like(pos)
        d_loss = tf.reduce_mean(tf.nn.relu(ones_x - pos)) + tf.reduce_mean(tf.nn.relu(ones_x + neg))
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss
