
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import time


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
  BB_EXTRA = 25

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

  def run(self, image, BB, filename):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """

    # Crop the input image based on previous frames' BB
    if (BB[0][0] == BB[1][0]):
        BB[0][0] = 0
        BB[1][0] = image.size[0]
    if (BB[0][1] == BB[1][1]):
        BB[0][1] = 0
        BB[1][1] = image.size[1]

    cropped_image = image.crop((BB[0][0], BB[0][1], BB[1][0], BB[1][1]))
    #image.save("./predictions_breakSF/" + filename + "image" + ".png")
    #cropped_image.save("./predictions_breakSF/" + filename + "cropped_image" + ".png")

    width, height = cropped_image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = cropped_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    pred = Image.fromarray(seg_map.astype(np.int32)).resize(cropped_image.size)
    #pred.save("./predictions_breakSF/" + filename + "pred" + ".png")
    # Pad zeros in the cropped regions to get the same shape as target label
    new_arr = np.zeros((image.size[1], image.size[0]))
    resized_pred = Image.fromarray(new_arr.astype(np.int32))
    resized_pred.paste(pred, (BB[0][0], BB[0][1], BB[1][0], BB[1][1]))
    new_BB = self.getBoundingBox(resized_pred)
    #resized_pred.save("./predictions_breakSF/" + filename + "resized_pred" + ".png")

    pred_map = np.array(resized_pred)
  
    return pred_map, new_BB


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
MODEL = DeepLabModel("../../train_DAVIS_FrozenGraph.tar.gz")
print('model loaded successfully!')

#%%
total_time = 0
for path, dirs, files in os.walk("./JPEGImages/480p/kid-football/"):
#for path, dirs, files in os.walk("./JPEGImages/480p/breakdance-flare/"):
    files.sort()
    frame_num = 1
    for filename in files:
        image = Image.open(path + "/" + filename)

        if frame_num == 1:
          gt_file = filename.replace(".jpg", ".png")
          gt_img = Image.open("./Annotations/kid-football/" + gt_file)
          gt_img = gt_img.resize(image.size)
          BB = MODEL.getBoundingBox(gt_img)

        start_time = time.time()
        seg_map, BB = MODEL.run(image, BB, filename)
        total_time += time.time() - start_time
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = Image.fromarray(seg_image).resize(image.size)
        blend = Image.blend(image, seg_image, alpha=0.7)

        #draw = ImageDraw.Draw(seg_image)
        #output = draw.rectangle(BB, outline=250)
        #del draw

        draw = ImageDraw.Draw(blend)
        output = draw.rectangle(BB, outline=250)
        del draw

        filename = filename[:-4]
        blend.save("./predictions_breakSF/" + filename + "blend" + ".png")
        #seg_image.save("./predictions/" + filename + ".png")

        frame_num += 1
        
print(total_time)
print(total_time / 100)
        
#        #%%
#gd = MODEL.sess.graph_def
##%%
#nodes = []
#for node in gd.node:
#    nodes.append((node.name, node.input))

