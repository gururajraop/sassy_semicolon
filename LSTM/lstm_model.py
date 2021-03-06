from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def rnn_model_fn(features, labels, mode):
  # Prepare inputs to be the right shape (batch_size. seq_length, 129,129, 256)
  # outside of this model.
  input_layer = features

  cell = tf.contrib.rnn.ConvLSTMCell(
          conv_ndims=2,
          input_shape=[129, 129, 256],
          output_channels=1,
          kernel_shape=[3,3]
      )
  
  initial_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
  LSTM_features = tf.nn.dynamic_rnn(cell=cell,
                                    inputs=x,
                                    initial_state=initial_state,
                                    dtype=tf.float32)[0][:,-1]
  
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
#  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#  train_data = mnist.train.images  # Returns np.array
#  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#  eval_data = mnist.test.images  # Returns np.array
#  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  

  inputs = np.random.random((4, 129, 129, 256))
  label = np.random.random((342, 513))
  
  # Create the Estimator
  LSTM = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/rnn_model")
  
  LSTM.predict()
#  # Set up logging for predictions
#  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "softmax_tensor"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)

#  # Train the model
#  train_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": train_data},
#      y=train_labels,
#      batch_size=100,
#      num_epochs=None,
#      shuffle=True)

#  gru.train(
#      input_fn=train_input_fn,
#      steps=20000,
#      hooks=[logging_hook])

  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
#  eval_results = gru.evaluate(input_fn=eval_input_fn)
#  print(eval_results)


if __name__ == "__main__":
  tf.app.run()