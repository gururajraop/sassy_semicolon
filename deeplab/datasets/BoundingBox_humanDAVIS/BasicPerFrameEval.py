#@title Imports
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

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


def get_IoU(prediction, labelImage):
  label = np.array(labelImage)

  TP = ((prediction == 1) & (label == 1)).sum()
  FP = ((prediction == 1) & (label != 1)).sum()
  FN = ((prediction != 1) & (label == 1)).sum()
  TN = ((prediction != 1) & (label != 1)).sum()

  if TP > 0 or FN > 0 or FP > 0: 
    IoU_human = TP / (TP + FN + FP)
  else:
    IoU_human = 1
  if TN > 0 or FP > 0 or FN > 0:
    IoU_bg = TN / (TN + FP + FN)
  else:
    IoU_bg = 1

  IoU = (IoU_human + IoU_bg) / 2

  return IoU

#%%

MODEL = DeepLabModel("../../train_COCO_FrozenGraph4.tar.gz")
print('model loaded successfully!')

#%%

val_set = open("./ImageSets/val.txt", "r").read().split('\n')

IoU = []
for val_case in val_set:
  class_IoU = []
  for _, _, files in os.walk("./Annotations/" + val_case):
    for segFile in files:
      imgFile = segFile.replace("png", "jpg")
      image = Image.open("./JPEGImages/480p/" + val_case + "/" + imgFile)
      image.getcolors()
      resized_im, seg_map = MODEL.run(image)

      label = Image.open("./Annotations/" + val_case + "/" + segFile)
      resized_label = label.resize(resized_im.size)
      iou_val = get_IoU(seg_map, resized_label)
      class_IoU.append(iou_val)
      #IoU.append(iou_val)

  current_mIOU = np.average(class_IoU)
  IoU.append(current_mIOU)
  print("Evaluation: class:{} , mIOU: {}".format(val_case, current_mIOU))

mIoU = np.average(IoU)
print("mIOU :", mIoU)
#%%
