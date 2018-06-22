import os
import glob
from PIL import Image


class DatasetConverter:
    # folder where the annotations of the humanDAVIS dataset are
    SOURCE_DAVIS_ANNOTATIONS_PATH = '../humanDAVIS/Annotations/'
    # folder where the images of the humanDAVIS dataset are
    SOURCE_DAVIS_IMAGES_PATH = '../humanDAVIS/JPEGImages/480p/'
    # folder where the splits of the humanDAVIS are
    SOURCE_SPLIT_PATH = '../humanDAVIS/ImageSets/'

    # folder where the converted annotations of the sf humanDAVIS dataset will be
    DEST_DAVIS_ANNOTATIONS_PATH = './Annotations/'
    # folder where the converted images of the sf humanDAVIS dataset will be
    DEST_DAVIS_IMAGES_PATH = './JPEGImages/480p/'
    # folder where the splits of the sf humanDAVIS dataset will be
    DEST_SPLIT_PATH = './ImageSets/'

    TRAIN = 'train.txt'
    VAL = 'val.txt'

    def convert_dataset(self):
        self.convert_subset(self.SOURCE_DAVIS_IMAGES_PATH, self.DEST_DAVIS_IMAGES_PATH)
        self.convert_subset(self.SOURCE_DAVIS_ANNOTATIONS_PATH, self.DEST_DAVIS_ANNOTATIONS_PATH)

    def convert_subset(self, source, dest):
        subdirectories = os.listdir(source)
        subdirectories.sort()

        for subdirectory in subdirectories:
            self.save_images(subdirectory, source, dest)


    def save_images(self, subdirectory, source_path, dest_path):
        source_path = source_path + subdirectory + '/'
        image_names = os.listdir(source_path)
        image_names.sort()

        for image_name in image_names:
            image = Image.open(source_path + image_name)
            new_image_name = subdirectory + '-' + image_name
            image.save(dest_path + new_image_name)

    def generate_splits(self):
        self.generate_split(self.TRAIN)
        self.generate_split(self.VAL)

    def generate_split(self, split):
        fd = open(self.SOURCE_SPLIT_PATH + split, 'r')
        subdirectories = fd.read()
        subdirectories = subdirectories.split('\n')

        final_image_names = []

        for subdirectory in subdirectories:
            image_names = glob.glob(self.DEST_DAVIS_IMAGES_PATH + subdirectory + '*.jpg' )
            new_image_names = [image_name[image_name.rindex('/')+1 : image_name.rindex('.')] for image_name in image_names]
            new_image_names.sort()
            final_image_names.extend(new_image_names)

        wd = open(self.DEST_SPLIT_PATH + split, 'w')
        wd.write("\n".join(final_image_names))




helper = DatasetConverter()
#helper.convert_dataset()
helper.generate_splits()
