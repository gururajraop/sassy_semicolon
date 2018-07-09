import numpy as np
import tensorflow as tf

BATCH_SIZE = 1
SEQUENCE_LENGTH = 4

# Just the size of some example segmentation output
OUTPUT_SIZE = (342, 513)

x = tf.placeholder(tf.float32, shape =(BATCH_SIZE, SEQUENCE_LENGTH, 129, 129, 256))
#y = tf.placeholder(tf.float32, shape =(342, 513))

cell = tf.contrib.rnn.ConvLSTMCell(
        conv_ndims=2,
        input_shape=[129, 129, 256],
        output_channels=1,
        kernel_shape=[5,5]
    )


initial_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
LSTM_features = tf.nn.dynamic_rnn(cell=cell,
                                  inputs=x,
                                  initial_state=initial_state,
                                  dtype=tf.float32)[0][:,-1]


# https://www.tensorflow.org/api_docs/python/tf/image/resize_images
resized_features = tf.image.resize_images(
    LSTM_features,
    OUTPUT_SIZE,
    method=tf.image.ResizeMethod.BICUBIC, # some other options (worse results but faster?)
    align_corners=False
)

squeezed_features = tf.squeeze(resized_features)


#%%

# Shape of features obtained at the end of deeplab's decoder (alwaus same shape)
inputs = np.random.random((BATCH_SIZE, SEQUENCE_LENGTH, 129, 129, 256))
label = np.random.random(OUTPUT_SIZE)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  output = sess.run(squeezed_features, feed_dict={x: inputs})
  


