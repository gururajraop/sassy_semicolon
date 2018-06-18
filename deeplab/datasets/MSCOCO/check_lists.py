import os

image_names = [os.path.splitext(x)[0] for x in os.listdir("./Images")]

print("Checking training set.")
labels = [x.strip('\n') for x in open("./Lists/train.txt", 'r')]
missing_train = []
for l in labels:
    if l not in image_names:
        missing_train.append(l)

print("Checking validation set.")

labels = [x.strip('\n') for x in open("./Lists/val.txt", 'r')]
missing_val = []

for l in labels:
    if l not in image_names:
        missing_val.append(l)