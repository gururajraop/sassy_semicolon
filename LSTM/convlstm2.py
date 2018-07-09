import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.rnn_cell import Conv2DLSTMCell

batch = 1
sequence = 2
H = 100
W = 100
C = 3

C2 = 30

def conv2d(input_, output_dim, ks=3, s=1, stddev=0.02, padding='SAME', name='conv2d'):
    import tensorflow.contrib.slim as slim
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name='deconv2d'):
    import tensorflow.contrib.slim as slim
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, [ks, ks], [s, s], padding='SAME')

def fake_network(inp):
    """
    Imagine our original architecture is:
    input -> conv2d -> deconv2d -> output segmentation

    Now we want to inject ConvLSTM somewhere in the middle. We inject ConvLSTM and add deconv2d on top
    to recover a segmenation image of the same size.
    """
    # Returns a fake intermediate feature
    assert inp.shape == (batch * sequence, H, W, C)

    # kernel size = 3, stride = 2
    inp2 = conv2d(inp, C2, ks=3, s=2)

    return inp2

input_ = tf.placeholder(
    tf.float32, [batch * sequence, H, W, C],
    "input_images")

out = fake_network(input_)

input_shape = [out.shape[1], out.shape[2], out.shape[3]]
out = tf.reshape(out, [batch, sequence] + input_shape)

# Returns again just 1 channel, in that case we just try to fuse predictions coming from "fake_network"
cell = Conv2DLSTMCell(input_shape=input_shape,
                      output_channels=out.shape[3],
                      kernel_shape=[5, 5],
                      use_bias=True,
                      name='conv_2d_lstm_cell_1')

# Set initial state to zero
initial_state = cell.zero_state(batch, dtype=tf.float32)

# rnn_features shape is [batch, sequence, H, W, 1]
rnn_features = tf.nn.dynamic_rnn(cell=cell,
                                 inputs=out,
                                 initial_state=initial_state,
                                 time_major=False,
                                 dtype=tf.float32,
                                 scope="rnn")[0]

# Get features for the last frame of the sequence!
# shape will be [batch, H, W, 1]
rnn_features = rnn_features[:,-1]

out = deconv2d(rnn_features, 1, ks=3, s=2)

# if you don't want to do that (when we have batch * sequence labels) we can do:
# rnn_features = tf.reshape(rnn_features, [batch * sequence, H, W, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    result = sess.run(out,
                      feed_dict={input_: np.random.random(size=[batch * sequence, H, W, 3]).astype(np.float32)})
    print(result.shape)