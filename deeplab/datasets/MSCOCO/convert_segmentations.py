import os
from PIL import Image
import json
import numpy as np

# Create 2D array with human colors set to 1, others to 0
def convert_from_color_segmentation(arr_3d, colors, human_colors):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for color in colors:
        m = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        
        if color in human_colors:
            arr_2d[m] = 1
        else:
            arr_2d[m] = 0   
    return arr_2d

# Find the corresponding segment for each color in the image using area values
def convert_annotations(annotations, path):
    if not os.path.exists(path):
        	os.makedirs(path)
    
    counter = 0
    counter2 = 0
    total = len(annotations)
    
    for annotation in annotations:
        counter += 1
        
        if counter % 100 == 0:
            print("%d images converted out of %d." % (counter, total))
        
        # Find the segments depicting humans and save the area values
        human_areas = [i["area"] for i in annotation["segments_info"] if i["category_id"] == 1]        
            
        if human_areas:
            human_colors = []
            title = annotation["file_name"]
            arr = Image.open("panoptic_val2017/" + title)
            areas, colors = zip(*arr.getcolors())
            
            # Handle ambiguous segments with the same area values
            duplicates = [x for x in areas if areas.count(x) > 1]
            
            if duplicates:
                counter2 += 1
                print("Two segmentations with the same area, count =", counter2, duplicates)
                duplicates = np.array(list(set(duplicates)))
                
                for dup in duplicates:
                    human_areas = [x for x in human_areas if x != dup]
                    indices_areas = np.where(areas == dup)[0]
                    segments = annotation["segments_info"]
                    indices_segments = np.where(np.array([segment['area'] for segment in segments]) == dup)[0]
                    
                    # Match color with the segment with the highest ratio of
                    # pixels in that color.
                    for idx_c in indices_areas:
                        best_ratio = 0
                        best_idx_s = 0
                        
                        for idx_s in indices_segments:
                            bbox = segments[idx_s]["bbox"]
                            boxed = arr.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                            ratio = np.all(np.array(boxed) == np.array(colors[idx_c]).reshape(1, 1, 3), axis=2).sum() / dup
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_idx_s = idx_s
                                
                        if segments[best_idx_s]["category_id"] == 1:
                            human_colors.append(colors[idx_c])
    
            arr = np.array(arr)
            
            for area in human_areas:
                human_colors.append(colors[areas.index(area)])
                
            image = convert_from_color_segmentation(arr, colors, human_colors)
            Image.fromarray(image).save(path + "/" + title)
            
if __name__ == '__main__':
    with open('panoptic_val2017.json') as f:
        data = json.load(f)

    annotations = data["annotations"]
    path = "./converted_val"
    
    convert_annotations(annotations, path)
    print("Converting validation set finished")
    
    with open('panoptic_train2017.json') as f:
        data = json.load(f)

    annotations = data["annotations"]
    path = "./converted_train"
    
    convert_annotations(annotations, path)
    print("Converting training set finished")
    


