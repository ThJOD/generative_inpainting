import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *

from SNConvolution import SNConvolution2D

logger = logging.getLogger()
np.random.seed(2018)


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x


#https://github.com/JiahuiYu/generative_inpainting/issues/62#issuecomment-398552261
@add_arg_scope
def gated_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number for each feature and gating convolution.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, 2 * cnum, ksize, stride, dilation_rate=rate, #Double the cnum, because of split later
        padding=padding, name=name + '_feature_gating') #here no activation, see paper
    x1,x2 = tf.split(x,num_or_size_splits=2,axis=3,name=name + '_split')
    if activation != None:
        x = tf.math.multiply(tf.math.sigmoid(x2),activation(x1),name=name)
    else:
        x = tf.math.multiply(tf.math.sigmoid(x2),x1,name=name)
    return x

@add_arg_scope
def gated_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor,dynamic=True)
        x = gated_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x

@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor,dynamic=True)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x




#https://github.com/pfnet-research/sngan_projection/blob/master/source/links/sn_convolution_2d.py
@add_arg_scope
def sn_conv(x, cnum, ksize=5, stride=2, name='conv', training=True,sn=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output
    """
    conv = SNConvolution2D(x, cnum, ksize, stride,  (1,1,1,1), name, training, 'SAME', 1)
    if sn:
        x = conv.conv()
    else:
        x = conv.conv_noSN()
    x = tf.nn.leaky_relu(x)
    return x


def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
    maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
    t = tf.random_uniform(
        [], minval=config.VERTICAL_MARGIN, maxval=maxt, dtype=tf.int32)
    l = tf.random_uniform(
        [], minval=config.HORIZONTAL_MARGIN, maxval=maxl, dtype=tf.int32)
    h = tf.constant(config.HEIGHT)
    w = tf.constant(config.WIDTH)
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask

def strokeMask(config, name='mask'):
    def generateMask(height, width, maxVertex, maxLength, maxBushWidth, maxAngle, numMasks= 4, mask=None):
        if mask is None:
            mask = np.zeros((height, width), np.float32)
        for i in range(numMasks):
            numVertex = np.random.randint(maxVertex / 2,maxVertex)
            startX = np.random.randint(0,width - 1)
            startY = np.random.randint(0,height - 1)
            for i in range(numVertex):
                angle = np.random.uniform(0.0,maxAngle)
                if( i % 2 == 0):
                    angle = 2 * np.pi - angle
                length = np.random.uniform(maxLength / 4,maxLength)
                brushWidth = np.random.randint(maxBushWidth / 4,maxBushWidth)
                #print(type(np.floor(startX + length * np.sin(angle))))
                cv2.line(mask,(startX,startY),(int(np.floor(startX + length * np.sin(angle))),int(np.floor(startY + length * np.cos(angle)))),1,brushWidth)
                startX = int(np.floor(startX + length * np.sin(angle)))
                startY = int(np.floor(startY + length * np.cos(angle)))
                cv2.circle(mask,(startX,startY),int(brushWidth / 2.0),1)
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 0 )
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 1 )
        mask = mask.reshape(1,height,width,1)
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        maxVertex = config.MAX_VERTEX
        maxLength = config.MAX_LENGTH
        maxBushWidth = config.MAX_BUSH_WIDTH
        maxAngle = config.MAX_ANGLE
        numMasks = config.NUM_STROKES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_function(generateMask,[height, width, maxVertex,maxLength,maxBushWidth,maxAngle,numMasks],tf.float32)
        #mask = generateMask(height, width, maxVertex,maxLength,maxBushWidth,maxAngle,numMasks)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize

def resize_mask_scale(mask, scale = 1./4.):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, scale = scale, dynamic = True,
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config.SPATIAL_DISCOUNTING_GAMMA
    shape = [1, config.HEIGHT, config.WIDTH, 1]
    if config.DISCOUNTED_MASK:
        logger.info('Use spatial discounting l1 loss.')
        mask_values = np.ones((config.HEIGHT, config.WIDTH))
        for i in range(config.HEIGHT):
            for j in range(config.WIDTH):
                mask_values[i, j] = max(
                    gamma**min(i, config.HEIGHT-i),
                    gamma**min(j, config.WIDTH-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME') 
    #b has shape [batch,height,width,channels] the output will have shape [batch,rows_of_patches, collumns_of_patches, kernel_width * kernel_width * channels] https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
    #rows_of_patches is how many rows in b can be used for patches. It depends on height and kernel_height and stride_height and the padding mode
    #collumns_of_patches is how many cols in b can be used for patches. It depends on width and kernel_width and stride_width and the padding mode
    
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    #f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    #b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor, dynamic=True)
    b = resize(b, scale=1./rate, func=tf.image.resize_nearest_neighbor, dynamic=True) #https://github.com/JiahuiYu/generative_inpainting/issues/194
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor, dynamic=True)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        #flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor, dynamic=True)
    return y, flow


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))


def roofMask(config, name='mask'):
    def generateMask(diagonal_length_mean, diagonal_length_deviation, cross_point_deviation, cross_angle_mean, cross_angle_deviation,height, width, mask=None):
        if mask is None:
            mask = np.zeros((height, width), np.float32)
        point1 = (np.random.randint(width//8,width - width//8),np.random.randint(height//4,height//2))
        quarter = np.random.randint(0,3)
        widthFactor = 1
        heightFactor = 1
        if quarter > 0:
            widthFactor *= -1
        if quarter % 2 == 0:
            heightFactor *= -1
        point2 = (np.random.randint(width//8,width - width//8) * widthFactor,np.random.randint(height//4,height//2) * heightFactor)
        diag1Length = np.linalg.norm(np.array(point1)-np.array(point2))
        crossPointDist = np.amax((np.random.normal(diag1Length//2, cross_point_deviation * diag1Length,1),0.0001))
        crossPointRelDistInv = diag1Length / crossPointDist
        crossPoint = np.array([(point1[0] + point2[0])//crossPointRelDistInv,(point1[1] + point2[1])//crossPointRelDistInv])
        #print(crossPoint)
        diag2Length = np.amax((np.random.normal(diag1Length, diag1Length * diagonal_length_deviation ,1),0.0))
        #print(diag2Length)
        crossAngle = np.random.normal(cross_angle_mean, cross_angle_deviation,1)
        #print(crossAngle)
        crossAngle += np.arctan((point1[1] - crossPoint[1]) / ((point1[0] - crossPoint[0]) +0.0000001))
        #print(crossAngle)
        point3 = (crossPoint[0] + diag2Length//2 * np.cos(crossAngle),np.amin((crossPoint[1] + diag2Length//2 * np.sin(crossAngle),height//2)))
        point4 = (crossPoint[0] + diag2Length//2 * np.cos(crossAngle + np.pi),np.amin((crossPoint[1] + diag2Length//2 * np.sin(crossAngle + np.pi),height//2)))
        points = np.array((point1, point2, point3, point4  ), dtype=int)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 1) 
        #cv2.line(mask,point1,point2,2,10)
        #cv2.line(mask,point3,point4,3,10)
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 0 )
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 1 )
        mask = mask.reshape(1,height,width,1)
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        diagonal_length_mean = config.DIAGONAL_LENGTH_MEAN
        diagonal_length_deviation = config.DIAGONAL_LENGTH_DEVIATION
        cross_point_deviation = config.CROSS_POINT_DEVIATION
        cross_angle_mean = config.CROSS_ANGLE_MEAN
        cross_angle_deviation = config.CROSS_ANGLE_DEVIATION
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_function(generateMask,[diagonal_length_mean, diagonal_length_deviation, cross_point_deviation, cross_angle_mean,cross_angle_deviation,height,width],tf.float32)
        #mask = generateMask(height, width, maxVertex,maxLength,maxBushWidth,maxAngle,numMasks)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def treeMask(config, name='mask'):
    def generateMask(height, width,stem_width_mean, stem_width_deviation, stem_height_mean, stem_height_deviation, crown_width_mean,crown_width_deviation, crown_height_mean, crown_height_deviation,  mask=None):
        if mask is None:
            mask = np.zeros((height, width), np.float32)
        startX = np.random.randint(0,width)
        stemWidth = int(np.amax((np.random.normal(stem_width_mean, stem_width_deviation,1),1)))
        stemHeight = int(np.amax((np.random.normal(stem_height_mean, stem_height_deviation,1),1)))
        cv2.line(mask,(startX,0),(startX, stemHeight),1, stemWidth)
        crownWidth = int(np.amax((np.random.normal(crown_width_mean, crown_width_deviation,1),1)))
        crownHeight = int(np.amax((np.random.normal(crown_height_mean, crown_height_deviation,1),1)))
        cv2.ellipse(mask,(startX,stemHeight - 10 + crownHeight//2),(crownWidth//2,crownHeight//2),0,0,360,1,-1)
        
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 0 )
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 1 )
        mask = mask.reshape(1,height,width,1)
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        stem_width_mean = config.STEM_WIDTH_MEAN
        stem_width_deviation = config.STEM_WIDTH_DEVIATION
        stem_height_mean = config.STEM_HEIGHT_MEAN
        stem_height_deviation = config.STEM_HEIGHT_DEVIATION
        crown_width_mean = config.CROWN_WIDTH_MEAN
        crown_width_deviation = config.CROWN_WIDTH_DEVIATION
        crown_height_mean = config.CROWN_HEIGHT_MEAN
        crown_height_deviation = config.CROWN_HEIGHT_DEVIATION
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_function(generateMask,[height,width,stem_width_mean,stem_width_deviation,stem_height_mean, stem_height_deviation, crown_width_mean,crown_width_deviation, crown_height_mean, crown_height_deviation ],tf.float32)
        #mask = generateMask(height, width, maxVertex,maxLength,maxBushWidth,maxAngle,numMasks)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def treeGroupMask(config, name='mask'):
    def generateMask(height, width,minTrees,maxTrees, crown_angle_mean, crown_angle_deviation, crown_width_mean,crown_width_deviation, crown_height_mean, crown_height_deviation,  mask=None):
        if mask is None:
            mask = np.zeros((height, width), np.float32)
        numTrees = np.random.randint(minTrees,maxTrees)
        xCenter = 0
        crownWidth = np.random.normal(crown_width_mean, crown_width_deviation,numTrees)
        crownHeight = np.random.normal(crown_height_mean, crown_height_deviation,numTrees)
        crownAngle = np.random.normal(crown_angle_mean, crown_angle_deviation,numTrees)
        for i in range(numTrees):
            cv2.ellipse(mask,(int(xCenter),0),(int(np.amax((crownWidth[i],1)))//2,int(np.amax((crownHeight[i],1)))//2),0,0,360,1,-1)
            #cv2.ellipse(mask,(int(xCenter),20),(25//2,80//2),0,0,360,1,-1)
            xCenter += crownWidth[i]//3
        
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 0 )
        if np.random.randint(0,2):
            mask = cv2.flip( mask, 1 )
        mask = mask.reshape(1,height,width,1)
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        crown_angle_mean = config.CROWN_ANGLE_MEAN
        crown_angle_deviation = config.CROWN_ANGLE_DEVIATION
        crown_width_mean = config.CROWN_GROUP_WIDTH_MEAN
        crown_width_deviation = config.CROWN_GROUP_WIDTH_DEVIATION
        crown_height_mean = config.CROWN_GROUP_HEIGHT_MEAN
        crown_height_deviation = config.CROWN_GROUP_HEIGHT_DEVIATION
        minTrees = config.MIN_TREES
        maxTrees = config.MAX_TREES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_function(generateMask,[height,width,minTrees, maxTrees,crown_angle_mean,crown_angle_deviation, crown_width_mean,crown_width_deviation, crown_height_mean, crown_height_deviation ],tf.float32)
        #mask = generateMask(height, width, maxVertex,maxLength,maxBushWidth,maxAngle,numMasks)
        mask.set_shape([1] + [height, width] + [1])
    return mask

def RandomMaskGenerator(config, name='mask'):
    maskType = np.random.randint(0,3)
    maskF = None
    if maskType == 0:
        maskF = roofMask
    if maskType == 1:
        maskF = treeMask
    if maskType == 2:
        maskF = treeGroupMask
    mask = maskF(config,name)
    return mask
        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)


