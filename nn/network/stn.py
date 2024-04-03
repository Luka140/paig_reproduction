from six.moves import xrange
import tensorflow as tf
import torch
import torch.nn.functional as F

def tf_stn(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.compat.v1.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones([n_repeats, ]), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.compat.v1.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            #x = tf.Print(x,[x],message="x: ", summarize=1000)

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f-1.01) / 2.0
            #x = tf.Print(x,[x],message="x_floor: ", summarize=1000)

            y = (y + 1.0)*(height_f-1.01) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            #x0 = tf.Print(x0,[x0],message="x0: ", summarize=1000)

            x1 = x0 + 1
            
            y0 = tf.cast(tf.floor(y), 'int32')
            #y0 = tf.Print(y0,[y0],message="y0: ", summarize=1000)

            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            #x0 = tf.Print(x0,[x0],message="x0_clip: ", summarize=1000)

            x1 = tf.clip_by_value(x1, zero, max_x)
            #x1 = tf.Print(x1,[x1],message="x1_clip: ", summarize=1000)

            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, [-1, channels])
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)

            """
            wa = tf.Print(wa,[wa],message="wa: ", summarize=1000)
            wb = tf.Print(wb,[wb],message="wb: ", summarize=1000)
            wc = tf.Print(wc,[wc],message="wc: ", summarize=1000)
            wd = tf.Print(wd,[wd],message="wd: ", summarize=1000)
            """

            a = wa*Ia
            b = wb*Ib
            c = wc*Ic
            d = wd*Id

            """
            a = tf.Print(a,[a],message="wa*Ia: ", summarize=1000)
            b = tf.Print(b,[b],message="wb*Ib: ", summarize=1000)
            c = tf.Print(c,[c],message="wc*Ic: ", summarize=1000)
            d = tf.Print(d,[d],message="wd*Id: ", summarize=1000)
            """

            output = tf.add_n([a,b,c,d])

            #output = tf.Print(output,[output],message="output1: ", summarize=1000)

            return output

    def _meshgrid(height, width):
        with tf.compat.v1.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=[height, 1]),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=[1, width]))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.compat.v1.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, [num_batch])
            grid = tf.reshape(grid, [num_batch, 3, -1])

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)

            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            #input_transformed = tf.Print(input_transformed,[input_transformed],message="input_transformed: ", summarize=1000)

            output = tf.reshape(
                input_transformed, [num_batch, out_height, out_width, num_channels])

            #output = tf.Print(output,[output],message="output: ", summarize=1000)

            return output

    with tf.compat.v1.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def tf_batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer
    Parameters
    ----------
    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]
    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with tf.compat.v1.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
    return transformer(input_repeated, thetas, out_size)

def stn(U, theta, out_size):
    U_pytorch = U.permute(0, 3, 1, 2).clone().detach()

    print("U", U.shape)
    print("U_pytorch", U_pytorch.shape)
    print("theta", theta.shape)

    num_batch = theta.size(0)
    num_transforms = theta.size(1)
    num_channels = U_pytorch.size(1)
    theta = theta.view(-1, 2, 3)  # Reshape theta to have shape (num_batch*num_transforms, 2, 3)
    print(theta.shape)
    # Adjust the size to match the expected dimensions
    size = torch.Size((num_batch, num_channels, *out_size))
    print("size=", size)
    grid = F.affine_grid(theta, size)
    output = F.grid_sample(U_pytorch, grid)

    return output.permute(0, 2, 3, 1)  # Transpose dimensions to match expected shape

def batch_transformer(U, thetas, out_size):
    U_pytorch = U.permute(0, 3, 1, 2).clone().detach()
    num_batch, num_transforms = thetas.shape[:2]
    input_repeated = U_pytorch.unsqueeze(1).repeat(1, num_transforms, 1, 1, 1)
    input_repeated = input_repeated.view(-1, *U_pytorch.shape[1:])
    return stn(input_repeated, thetas, out_size)