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
    """ Comptes the Bounding-Box for the provided segmentation map

    Args:
      image: Input segmentation map for which the BB nees to be found

    Returns:
      xy_coor: Returns the Bounding-Box for the segmentation map
    """

    u = np.where(np.array(image) == 1)
    #u = np.where(image == 1)
    if len(u[1]) != 0:
      x_min = max(np.min(u[1]) - self.BB_EXTRA, 0)
      x_max = min(np.max(u[1]) + self.BB_EXTRA, image.size[0])
    else:
      x_min = 0
      x_max = image.size[0]

    if len(u[0]) != 0:
      y_min = max(np.min(u[0]) - self.BB_EXTRA, 0)
      y_max = min(np.max(u[0]) + self.BB_EXTRA, image.size[1])
    else:
      y_min = 0
      y_max = image.size[1]

    xy_coor = [(x_min, y_min), (x_max, y_max)]
    return xy_coor

  def run(self, org_image, BB, mode):
    """Runs inference on a single image.

    Args:
      org_image: A PIL.Image object, raw input image.
      BB       : The Bounding-Box from the previous frame
      mode     : Mode of the segmentation image

    Returns:
      resized_pred: Segmentation map of resized prediction from the deeplab
      new_BB      : Calculated Bounding-Box of the current frame based on the resized_pred
    """

    # Crop the input image based on previous frames' BB
    if (BB[0][0] == BB[1][0]):
        BB[0][0] = 0
        BB[1][0] = org_image.size[0]
    if (BB[0][1] == BB[1][1]):
        BB[0][1] = 0
        BB[1][1] = org_image.size[1]

    cropped_image = org_image.crop((BB[0][0], BB[0][1], BB[1][0], BB[1][1]))

    # Get the predictions using deeplab
    width, height = cropped_image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (max(int(resize_ratio * width), 1), max(int(resize_ratio * height), 1))
    resized_image = cropped_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    # Convert the prediction to image type
    pred = Image.fromarray(seg_map.astype(np.int32)).resize(cropped_image.size)

    # Pad zeros in the cropped regions to get the same shape as target label
    new_arr = np.zeros((org_image.size[1], org_image.size[0]))
    resized_pred = Image.fromarray(new_arr.astype(np.int32))
    resized_pred.paste(pred, (BB[0][0], BB[0][1], BB[1][0], BB[1][1]))
    new_BB = self.getBoundingBox(resized_pred)

    return resized_pred, new_BB


def get_IoU(predictionImage, labelImage):
  """ Compute the mIoU value for the predicition

  Args:
    predictionImage: Prediction from the network
    labelImage     : Ground truth of the image

  Returns:
    mIoU: The mean-Intersection-Over-Union value for the prediction
  """
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

  IoU = (IoU_human + IoU_bg) / 2
  #IoU = ((0.5 * IoU_human) + (1.5 * IoU_bg)) / 2

  return IoU

#%%

#MODEL = DeepLabModel("../../train_COCO_FrozenGraph4.tar.gz")
#MODEL = DeepLabModel("../../train_DAVIS_FrozenGraph.tar.gz")
MODEL = DeepLabModel("../../train_COCO_MobileNetV3.tar.gz")
print('model loaded successfully!')

#%%

val_set = open("./ImageSets/val.txt", "r").read().split('\n')

IoU = []
# Go through all the videos of the validation set
for val_case in val_set:
  class_IoU = []

  # Initialize some values
  frame_id = 1
  for _, _, files in os.walk("./JPEGImages/480p/" + val_case):
    files.sort()
    for imgFile in files:
      image = Image.open("./JPEGImages/480p/" + val_case + "/" + imgFile)

      # Get the ground truth for the first frame and generated Bounding-Box using that
      # This is done only for first frame and not for any other subsequent frames
      if frame_id == 1:
        segFile = imgFile.replace("png", "jpg")
        label = Image.open("./Annotations/" + val_case + "/" + segFile)
        BB = MODEL.getBoundingBox(label.resize(image.size))

      # Run the forward prediciton and obtain the prediction and BB for the current frame
      pred, BB = MODEL.run(image, BB, label.mode)

      resized_label = label.resize(pred.size)

      # Calculate the mIoU for the frame
      iou_val = get_IoU(pred, resized_label)
      class_IoU.append(iou_val)
      #IoU.append(iou_val)

      frame_id += 1

  current_mIOU = np.average(class_IoU)
  IoU.append(current_mIOU)
  print("Evaluation: class:{}, mIOU: {}".format(val_case, current_mIOU))

# Get the final mIoU for all the vidoes in the val set
mIoU = np.average(IoU)
print("mIOU :", mIoU)
#%%
