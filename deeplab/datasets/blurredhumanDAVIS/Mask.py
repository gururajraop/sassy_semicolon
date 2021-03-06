#@title Imports
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time

import numpy as np
from PIL import Image
from PIL import ImageFilter

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

  def blurImage(self, image, mask):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=3))
    blurred_image = self.unblur_mask(blurred_image, image, mask)

    return blurred_image

  def unblur_mask(self, blurred_image, image, mask):
    bi_data = blurred_image.load()
    i_data = image.load()
    m_data = mask.load()

    for y in range(image.size[1]):
      for x in range(image.size[0]):
        if m_data[x, y] == 1:
          bi_data[x, y] = i_data[x, y]

    return blurred_image

  def run(self, org_image, mask):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """

    # Do not blur for first frame
    if (mask):
      image = self.blurImage(org_image, mask)
    else:
      image = org_image

    # Get the predictions using deeplab
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)


    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    # Convert the prediction to image type
    pred = Image.fromarray(seg_map.astype(np.int32))

    return pred.resize(image.size)


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
RESULTS = './Results/'
MODEL = DeepLabModel("../../train_DAVIS_Blurred_FrozenGraph.tar.gz")
print('model loaded successfully!')

#%%

val_set = open("../BoundingBox_humanDAVIS/ImageSets/val.txt", "r").read().split('\n')

IoU = []
start = time.time()
num = 0
for val_case in val_set:
  class_IoU = []
  mask = None
  result_folder = RESULTS + val_case + '/'

  if not os.path.exists(result_folder):
    os.makedirs(result_folder)

  for _, _, files in os.walk("../BoundingBox_humanDAVIS/Annotations/" + val_case):
    files.sort()
    for segFile in files:
      imgFile = segFile.replace("png", "jpg")
      image = Image.open("../BoundingBox_humanDAVIS/JPEGImages/480p/" + val_case + "/" + imgFile)
      label = Image.open("../BoundingBox_humanDAVIS/Annotations/" + val_case + "/" + segFile)
      image.getcolors()

      mask = MODEL.run(image, mask)
      colors = mask.getcolors
      mask.save(result_folder + segFile)

      resized_label = label.resize(mask.size)

      iou_val = get_IoU(mask, resized_label)
      class_IoU.append(iou_val)
      #IoU.append(iou_val)
      num += 1

  current_mIOU = np.average(class_IoU)
  IoU.append(current_mIOU)
  print("Evaluation: class:{} , mIOU: {}".format(val_case, current_mIOU))

mIoU = np.average(IoU)
print("mIOU :", mIoU)

end = time.time()
print("Total time:", (end-start)/num)
#%%
