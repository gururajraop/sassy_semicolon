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
  BB_EXTRA = 10

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

  def getBoundingBox(self, image):
    u = np.where(np.array(image) == 1)
    #u = np.where(image == 1)
    if len(u[1]) != 0:
      x_min = min(np.min(u[1]) - self.BB_EXTRA, 0)
      x_max = min(np.max(u[1]) + self.BB_EXTRA, image.size[1])
    else:
      x_min = 0
      x_max = image.size[1]

    if len(u[0]) != 0:
      y_min = min(np.min(u[0]) - self.BB_EXTRA, 0)
      y_max = min(np.max(u[0]) + self.BB_EXTRA, image.size[0])
    else:
      y_min = 0
      y_max = image.size[0]

    xy_coor = [(x_min, y_min), (x_max, y_max)]
    return xy_coor

  def run(self, org_image, BB, mode, pred_size):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    #if (BB is None):
    #  BB = [(0,0), org_image.size]

    image = org_image#.crop((BB[0][0], BB[0][1], BB[1][0], BB[1][1]))

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    pred = Image.fromarray(seg_map.astype(np.int32))
    if (BB is None) or (pred_size is None):
      new_BB = self.getBoundingBox(pred)
      resized_pred = pred
    else:
      pred = pred.resize(pred_size)
      cropped_pred = pred.crop((BB[0][0], BB[0][1], BB[1][0], BB[1][1]))
      resized_pred = Image.new(mode, pred_size)
      resized_pred.paste(cropped_pred, (BB[0][0], BB[0][1], BB[1][0], BB[1][1]))
      new_BB = self.getBoundingBox(resized_pred)

    return resized_pred, resized_pred.size, new_BB


def get_IoU(predictionImage, labelImage):
  prediction = np.array(predictionImage)
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

  #IoU = (IoU_human + IoU_bg) / 2
  IoU = ((0.5 * IoU_human) + (1.5 * IoU_bg)) / 2

  return IoU

#%%

MODEL = DeepLabModel("../../train_COCO_FrozenGraph4.tar.gz")
print('model loaded successfully!')

#%%

val_set = open("./ImageSets/val.txt", "r").read().split('\n')

IoU = []
for val_case in val_set:
  class_IoU = []
  BB = None
  pred_size = None
  for _, _, files in os.walk("./Annotations/" + val_case):
    for segFile in files:
      imgFile = segFile.replace("png", "jpg")
      image = Image.open("./JPEGImages/480p/" + val_case + "/" + imgFile)
      label = Image.open("./Annotations/" + val_case + "/" + segFile)
      image.getcolors()
      seg_map, pred_size, BB = MODEL.run(image, BB, label.mode, pred_size)

      resized_label = label.resize(pred_size)

      iou_val = get_IoU(seg_map, resized_label)
      class_IoU.append(iou_val)
      #IoU.append(iou_val)

  current_mIOU = np.average(class_IoU)
  IoU.append(current_mIOU)
  print("Evaluation: class:{} , mIOU: {}".format(val_case, current_mIOU))

mIoU = np.average(IoU)
print("mIOU :", mIoU)
#%%
