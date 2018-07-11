# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts humanDAVIS data to TFRecord file format with Example protos.

humanDAVIS dataset is expected to have the following directory structure:

  + datasets
    - build_data.py
    - build_humanDAVIS_data.py (current working directory).
    + humanDAVIS
      + JPEGImages
      + Lists
      + ConvertedAnnotations
      + tfrecord

Image folder:
  ./humanDAVIS/JPEGImages

Semantic segmentation annotations:
  ./humanDAVIS/Annotations

list folder:
  ./humanDAVIS/ImageSets

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os
import sys
import build_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS.image_format = 'jpg'

tf.app.flags.DEFINE_string('image_folder',
                           './BoundingBox_humanDAVIS/JPEGImages/480p/',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './BoundingBox_humanDAVIS/Annotations',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './BoundingBox_humanDAVIS/ImageSets',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './BoundingBox_humanDAVIS/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  subdirectories = [x.strip('\n') for x in open(dataset_split, 'r')]

  _NUM_SHARDS = len(subdirectories)

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    subdirectory = subdirectories[shard_id]
    image_names = os.listdir(FLAGS.image_folder +  subdirectory)
    filenames = [subdirectory + '/' + image_name[:image_name.rindex('.')] for image_name in image_names]
    filenames.sort()

    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

      for i in range(len(filenames)):
        sys.stdout.write('\r>> Converting subdirectory %s shard %d' % (subdirectory, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
