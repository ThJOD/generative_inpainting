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
