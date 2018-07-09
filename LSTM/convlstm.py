import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.rnn_cell import Conv2DLSTMCell

batch = 1
sequence = 2
H = 100
W = 100
C = 3

def fake_network(inp):
    # Returns a fake segmentation prediction
    assert inp.shape == (batch * sequence, H, W, C)

    return tf.zeros([batch * sequence, H, W, 1])

input_ = tf.placeholder(
    tf.float32, [batch * sequence, H, W, C],
    "input_images")

out = fake_network(input_)

out = tf.reshape(out, [batch, sequence, H, W, 1])

# Returns again just 1 channel, in that case we just try to fuse predictions coming from "fake_network"
cell = Conv2DLSTMCell(input_shape=[H, W, 1],
                      output_channels=1,
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

# if you don't want to do that (when we have batch * sequence labels) we can do:
# rnn_features = tf.reshape(rnn_features, [batch * sequence, H, W, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    result = sess.run(rnn_features,
                      feed_dict={input_: np.random.random(size=[batch * sequence, H, W, 3]).astype(np.float32)})
    print(result.shape)
