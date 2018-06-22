import os
from PIL import Image

class AnnotationsModifier:
    # folder where the annotations of the original DAVIS dataset are
    SOURCE_DAVIS_ANNOTATIONS_PATH = './Annotations/'
    # folder where the converted annotations of the DAVIS dataset will be
    DEST_DAVIS_ANNOTATIONS_PATH = './ConvertedAnnotations/'

    # colors in DAVIS pallete
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    PURPLE = 5
    PASTEL_BLUE = 6
    GRAY = 7
    BROWN = 8

    # color in which humans will be recolored
    HUMAN_COLOR = RED

    def modify_all_annotations(self):
        subdirectories = os.listdir(self.SOURCE_DAVIS_ANNOTATIONS_PATH)
        subdirectories.sort()

        for subdirectory in subdirectories:
            self.call_method(subdirectory)

    def call_method(self, subdirectory):
        method_name = 'modify_' + subdirectory.replace('-', '_') + '_' + 'annotations'
        getattr(self, method_name)(subdirectory)

    def create_dest_folder(self, subdirectory):
        path = self.DEST_DAVIS_ANNOTATIONS_PATH + subdirectory + '/'

        if not os.path.exists(path):
            os.makedirs(path)

    def get_images(self, subdirectory):
        image_files = []
        image_colors = []

        path = self.SOURCE_DAVIS_ANNOTATIONS_PATH + subdirectory + '/'
        image_names = os.listdir(path)
        image_names.sort()

        for image_name in image_names:
            image = Image.open(path + image_name)
            image_files.append(image)

            colors_info = image.getcolors()
            new_colors = [color[1] for color in colors_info if color[1] != self.BLACK]
            image_colors = list(set(image_colors + new_colors))

            #rgb_colors = image.convert('RGB').getcolors() # used to map pallete indices to pallete colors

        return image_names, image_files, image_colors

    def save_images(self, image_names, images, subdirectory):
        path = self.DEST_DAVIS_ANNOTATIONS_PATH + subdirectory + '/'

        for index in range(len(images)):
            image = images[index]
            image_name = image_names[index]
            image.save(path + image_name)

    def remove_color(self, image, color):
        pixdata = image.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if pixdata[x,y] == color:
                    pixdata[x,y] = self.BLACK

    def recolor_images(self, images, colors, human_colors):
        return_images = []

        for image in images:
            for color in colors:
                if color not in human_colors:
                    self.remove_color(image, color)

            for color in human_colors:
                if color != self.HUMAN_COLOR:
                    self.recolor_human(image, color)
            return_images.append(image)

        return return_images

    def recolor_human(self, image, color):
        pixdata = image.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if pixdata[x,y] == color:
                    pixdata[x,y] = self.HUMAN_COLOR

    def modify_annotations(self, subdirectory, human_colors):
        self.create_dest_folder(subdirectory)

        image_names, images, colors = self.get_images(subdirectory)
        images = self.recolor_images(images, colors, human_colors)

        self.save_images(image_names, images, subdirectory)

    def modify_bike_packing_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_bmx_bumps_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_bmx_trees_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)


    def modify_boxing_fisheye_annotations(self, subdirectory):
        human_colors = [self.GREEN, self.RED, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)


    def modify_breakdance_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_breakdance_flare_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_cat_girl_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_color_run_annotations(self, subdirectory):
        human_colors = [self.RED, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_crossing_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_dance_jump_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_dance_twirl_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_dancing_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_disc_jockey_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_drone_annotations(self, subdirectory):
        human_colors = [self.YELLOW, self.PURPLE]
        self.modify_annotations(subdirectory, human_colors)

    def modify_hike_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_hockey_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_horsejump_high_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_horsejump_low_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_india_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_judo_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_kid_football_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_kite_surf_annotations(self, subdirectory):
        human_colors = [self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_kite_walk_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_lab_coat_annotations(self, subdirectory):
        human_colors = [self.YELLOW, self.BLUE, self.PURPLE]
        self.modify_annotations(subdirectory, human_colors)

    def modify_lady_running_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_lindy_hop_annotations(self, subdirectory):
        human_colors = [self.RED, self.BLUE, self.GREEN, self.YELLOW, self.PURPLE, self.BROWN, self.GRAY, self.PASTEL_BLUE]
        self.modify_annotations(subdirectory, human_colors)

    def modify_loading_annotations(self, subdirectory):
        human_colors = [self.RED, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_longboard_annotations(self, subdirectory):
        human_colors = [self.BLUE, self.PURPLE]
        self.modify_annotations(subdirectory, human_colors)

    def modify_mbike_trick_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_miami_surf_annotations(self, subdirectory):
        human_colors = [self.RED, self.YELLOW, self.PURPLE]
        self.modify_annotations(subdirectory, human_colors)

    def modify_motocross_bumps_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_motocross_jump_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_motorbike_annotations(self, subdirectory):
        human_colors = [self.GREEN, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_paragliding_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_paragliding_launch_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_parkour_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_rollerblade_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_schoolgirls_annotations(self, subdirectory):
        human_colors = [self.RED, self.YELLOW, self.PURPLE, self.GRAY]
        self.modify_annotations(subdirectory, human_colors)

    def modify_scooter_black_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_scooter_board_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_scooter_gray_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_shooting_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_skate_park_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_snowboard_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_soapbox_annotations(self, subdirectory):
        human_colors = [self.GREEN, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_stroller_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_stunt_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_swing_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_tennis_annotations(self, subdirectory):
        human_colors = [self.RED]
        self.modify_annotations(subdirectory, human_colors)

    def modify_tractor_sand_annotations(self, subdirectory):
        human_colors = [self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_tuk_tuk_annotations(self, subdirectory):
        human_colors = [self.GREEN, self.RED, self.YELLOW]
        self.modify_annotations(subdirectory, human_colors)

    def modify_upside_down_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN]
        self.modify_annotations(subdirectory, human_colors)

    def modify_walking_annotations(self, subdirectory):
        human_colors = [self.RED, self.GREEN]
        self.modify_annotations(subdirectory, human_colors)


helper = AnnotationsModifier()
helper.modify_all_annotations()
