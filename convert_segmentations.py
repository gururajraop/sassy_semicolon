import os
from PIL import Image
import json
import numpy as np

def convert_from_color_segmentation(arr_3d, colors, human_colors):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for color in colors:
        m = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        if color in human_colors:
            arr_2d[m] = 150
        else:
            arr_2d[m] = 0   
    return arr_2d


with open('panoptic_val2017.json') as f:
    data = json.load(f)

annotations = data["annotations"]

newpath = "./converted_val"
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
counter = 0
total = len(annotations)

for annotation in annotations:
    counter += 1
    if counter % 100 == 0:
        print("%d images converted out of %d." % (counter, total))
        
    human_areas = [i["area"] for i in annotation["segments_info"] if i["category_id"] == 1]
        
    if human_areas:  
        title = annotation["file_name"]
        arr = Image.open("panoptic_val2017/" + title)
        areas, colors = zip(*arr.getcolors())
        arr = np.array(arr)
        
        human_colors = [] 
        for area in human_areas:
            human_colors.append(colors[areas.index(area)])
            
        image = convert_from_color_segmentation(arr, colors, human_colors)
        Image.fromarray(image).save("converted_val/" + title)