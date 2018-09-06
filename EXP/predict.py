
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time

#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2

import tensorflow as tf

#%%

#@title Helper methods


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
#  OUTPUT_TENSOR_NAME = 'decoder/decoder_conv1_pointwise/Relu:0'
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
    start = time.time()
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    model_time = time.time() - start
    seg_map = batch_seg_map[0]
    return resized_image, seg_map, model_time


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


BB_EXTRA = 0
def getBoundingBox(image):
    u = np.where(np.array(image)[:,:,0] == 128)
    #u = np.where(image == 1)
    if len(u[1]) != 0:
      x_min = max(np.min(u[1]) - BB_EXTRA, 0)
      x_max = min(np.max(u[1]) + BB_EXTRA, image.size[0])
    else:
      x_min = 0
      x_max = image.size[1]

    if len(u[0]) != 0:
      y_min = max(np.min(u[0]) - BB_EXTRA, 0)
      y_max = min(np.max(u[0]) + BB_EXTRA, image.size[1])
    else:
      y_min = 0
      y_max = image.size[0]

    xy_coor = [(x_min, y_min), (x_max, y_max)]
    return xy_coor




LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#%%

#MODEL = DeepLabModel("../../train_COCO_FrozenGraph4.tar.gz")
#MODEL = DeepLabModel("../../train_DAVIS_Blurred_FrozenGraph.tar.gz")
MODEL = DeepLabModel("../train_DAVIS_FrozenGraph.tar.gz")
print('model loaded successfully!')

#%%
model_time = 0
total_time = 0
#for frame_id in range(10):
frame_id = 0
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #image = Image.open("./Source1/Demo_" + str(frame_id) + ".jpg")
    ret, frame = cap.read()
    image = Image.fromarray(frame)
    if cap is None:
        continue
    start_time = time.time()
    resized_im, seg_map, deeplab_time = MODEL.run(image)
    #total_time += time.time() - start_time
    #model_time += deeplab_time
    total_time = time.time() - start_time
    model_time = deeplab_time
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = Image.fromarray(seg_image).resize(resized_im.size)
    blend = Image.blend(resized_im, seg_image, alpha=0.7)

    BB = getBoundingBox(seg_image)
    #draw = ImageDraw.Draw(seg_image)
    #output = draw.rectangle(BB, outline=250)
    #del draw

    #BB = getBoundingBox(blend)
    draw = ImageDraw.Draw(blend)
    output = draw.rectangle(BB, outline=250)
    del draw

    blend.save("./Result/Demo_" + str(frame_id) + ".png")
    #cv2.imshow('image', np.array(blend))
    #cv2.waitKey(1)
    frame_id += 1

    print("Xception model: FPS={0:0.2f}\r".format(1/model_time), end='', flush=True)
    #print(total_time)
    #print("\n\n")

print(model_time)        
print(total_time)
        
#        #%%
#gd = MODEL.sess.graph_def
##%%
#nodes = []
#for node in gd.node:
#    nodes.append((node.name, node.input))

