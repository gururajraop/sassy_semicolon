import numpy as np
import os
import sys
import json

person_file = open("./MSCOCO-humanIDs-train.txt", "r")

items = person_file.read().split('.')

ids = [int(s) for s in items[0].split(',')]
print(len(ids))

json_file = "./annotations/instances_train2017.json"
with open(json_file) as json_data:
	dataset = json.load(json_data)

for key, val in dataset.items():
	if key == 'images':
		print(len(val))
		images = [details['file_name'] for details in val if details['id'] in ids]
		print(len(images))

image_path = "./train2017/"
human_data = "./humanData/"
for image in images:
	image_full_path = image_path + str(image)
	cmd = 'cp ' + image_full_path + ' ' + human_data 
	os.system(cmd)
	

