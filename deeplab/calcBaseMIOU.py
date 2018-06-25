#@title Imports

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

#%%

#@title Helper methods


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def get_IoU(prediction, labelImage):
  label = np.array(labelImage)

  TP = ((prediction == 15) & (label == 1)).sum()
  FP = ((prediction == 15) & (label != 1)).sum()
  TN = ((prediction != 15) & (label == 1)).sum()
  FN = ((prediction != 15) & (label != 1)).sum()

  IoU_human = TP / (TP + FP + TN)
  IoU_bg = FN / (FN + FP + TN)

  IoU = (IoU_human + IoU_bg) / 2

  return IoU

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#%%

MODEL = DeepLabModel("../../deeplabv3_pascal_trainval_2018_01_04.tar.gz")
print('model loaded successfully!')

#%%

for path, dirs, files in os.walk("./datasets/MSCOCO/SegmentedImages"):
  IoU = []
  progress = 1
  total = len(files)
  for segFile in files:
    imgFile = segFile.replace("png", "jpg")
    image = Image.open("./datasets/MSCOCO/Images/" + imgFile)
    image.getcolors()
    resized_im, seg_map = MODEL.run(image)

    label = Image.open("./datasets/MSCOCO/SegmentedImages/" + segFile)
    resized_label = label.resize(resized_im.size)
    IoU.append(get_IoU(seg_map, resized_label))

    if (progress % 100 == 0):
      current_mIOU = np.average(IoU)
      print("Evaluation: {}/{} , mIOU: {}".format(progress, total, current_mIOU))

    progress += 1

  mIoU = np.average(IoU)
  print("mIOU :", mIoU)
#%%
