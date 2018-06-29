import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter


class BlurredDatasetConverter:
    # folder where the annotations of the humanDAVIS dataset are
    SOURCE_DAVIS_ANNOTATIONS_PATH = '../humanDAVIS/Annotations/'
    # folder where the images of the humanDAVIS dataset are
    SOURCE_DAVIS_IMAGES_PATH = '../humanDAVIS/JPEGImages/480p/'

    # folder where the converted images of the blurredhumanDAVIS dataset will be
    DEST_DAVIS_IMAGES_PATH = './JPEGImages/480p/'

    DEST_DAVIS_ANNOTATIONS_PATH = './Annotations/'

    VAL_FOLDERS = './ImageSets/val.txt'

    '''
    Only blur folders for training
    '''
    def blur_dataset(self):
        subdirectories = os.listdir(self.SOURCE_DAVIS_IMAGES_PATH)
        subdirectories.sort()

        fd = open(self.VAL_FOLDERS, 'r')
        val_subdirectories = fd.read()
        val_subdirectories = val_subdirectories.split('\n')

        for subdirectory in subdirectories:
            if subdirectory not in val_subdirectories:
                self.blur_images(subdirectory)
                print("Finished " + subdirectory)

    def get_images(self, subdirectory):
        input_images = []
        annotations = []

        input_path = self.SOURCE_DAVIS_IMAGES_PATH + subdirectory + '/'
        annotations_path = self.SOURCE_DAVIS_ANNOTATIONS_PATH + subdirectory + '/'

        input_image_names = os.listdir(input_path)
        input_image_names.sort()

        annotations_image_names = [image_name[:image_name.rindex('.')] + '.png' for image_name in input_image_names]

        for index in range(len(input_image_names)):
            input_image_name = input_image_names[index]
            annotations_image_name = annotations_image_names[index]

            image = Image.open(input_path + input_image_name)
            input_images.append(image)

            annotation = Image.open(annotations_path + annotations_image_name)
            annotations.append(annotation)

        return input_images, annotations, input_image_names, annotations_image_names

    def create_dest_folder(self, subdirectory):
        path = self.DEST_DAVIS_IMAGES_PATH + subdirectory + '/'
        path2 = self.DEST_DAVIS_ANNOTATIONS_PATH + subdirectory + "/"

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path2):
            os.makedirs(path2)

    def blur_images(self, subdirectory):
        self.create_dest_folder(subdirectory)
        input_images, annotations, image_names, annotations_image_names = self.get_images(subdirectory)
        blurred_images = [input_images[0]]

        for index in range(1, len(input_images)):
            image = input_images[index]
            previous_mask = annotations[index - 1]

            blurred_image = input_images[index].filter(ImageFilter.GaussianBlur(radius=3))
            blurred_image = self.unblur_mask(blurred_image, image, previous_mask)
            blurred_images.append(blurred_image)

        self.save_images(blurred_images, image_names, self.DEST_DAVIS_IMAGES_PATH + subdirectory + "/")
        self.save_images(annotations, annotations_image_names, self.DEST_DAVIS_ANNOTATIONS_PATH + subdirectory + "/")

    def save_images(self, blurred_images, image_names, path):

        for index in range(len(blurred_images)):
            image = blurred_images[index]
            image_name = image_names[index]
            image.save(path + image_name)

    def unblur_mask(self, blurred_image, image, mask):
        bi_data = blurred_image.load()
        i_data = image.load()
        m_data = mask.load()

        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if m_data[x, y] == 1:
                    bi_data[x, y] = i_data[x, y]

        return blurred_image

helper = BlurredDatasetConverter()
helper.blur_dataset()
