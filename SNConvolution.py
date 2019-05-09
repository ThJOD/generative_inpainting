import tensorflow as tf



def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter (fully differentiable version)
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    if u is None:
        u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    _u = u
    for _ in range(Ip):
        _v = tf.nn.l2_normalize(tf.matmul(_u, tf.transpose(W)))#Different to github, because W is transposed here
        _u = tf.nn.l2_normalize(tf.matmul(_v, W))
    _u = tf.stop_gradient(_u)
    _v = tf.stop_gradient(_v)
    sigma = tf.matmul(tf.matmul(_v,W),tf.transpose(_u))
    return sigma, _u, _v


#https://github.com/pfnet-research/sngan_projection/blob/master/source/links/sn_convolution_2d.py
class SNConvolution2D():
    
    def __init__(self,x, cnum, ksize=5, stride=(1,1,1,1), dilation=(1,1,1,1), name='conv', training=True, padding='SAME', Ip=1):
        """Define conv for discriminator.
        Activation is set to leaky_relu.

        Args:
            x: Input.
            cnum: Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            name: Name of layers.
            training: If current graph is for training or inference, used for bn.
            Ip: Iterations for weight parameter
        Returns:
            tf.Tensor: output

        """
        self.input = x
        self.output_channels = cnum
        self.ksize = ksize
        self.dilation = dilation
        self.name = name
        self.Ip = Ip
        self.w = tf.get_variable(name + "_kernel", shape=[ksize, ksize, x.get_shape()[-1], cnum], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),regularizer=None)
        self.b = tf.get_variable(name + "_bias", [cnum], initializer=tf.constant_initializer(0.0))
        self.u = tf.get_variable(name + "_u", [1, cnum], initializer=tf.random_normal_initializer(), trainable=False)
        self.stride = stride
        self.padding = padding

    def W_bar(self):
        """
        Spectrally Normalized Weight
        """
        w_shape = self.w.shape.as_list()
        W_mat = tf.reshape(self.w, [-1, w_shape[-1]]) #Produces a Matrix of shape [Inchannel * kernelW * kernelH, OutChannel]
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = tf.broadcast_to(tf.reshape(sigma,[1,1,1,1]),self.w.shape)
        return self.w / sigma

    def conv(self):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        assert self.padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            p = int(1*(self.ksize-1)/2)
            self.input = tf.pad(self.input , [[0,0], [p, p], [p, p], [0,0]], mode=self.padding)
            padding = 'VALID'
        x = tf.nn.conv2d(input=self.input, filter=self.W_bar(), strides=self.stride, dilations=self.dilation, padding=self.padding, name=self.name)
        return x