# Preprocess dataset to make data loading easier.
import shutil
import sys
import os

# The arguements are the path to the dataset
# First argument is the path to unzipped data
# Second arguement is path to empty directory where the the processed data would be stored.
# Usage example : python3 make_dataset.py images_background background
# Usage example : python3 make_dataset.py images_evaluation evaluation
data_path_read = sys.argv[1]
data_path_write = sys.argv[2]

for alphabeta in os.listdir(data_path_read):
    alphabeta_path = os.path.join(data_path_read, alphabeta)
    path_write1 = data_path_write[:-2] + '-' + alphabeta
    for charactor in os.listdir(alphabeta_path):
        charactor_path = os.path.join(alphabeta_path, charactor)
        path_write2 = path_write1 + '-' + charactor
        if not os.path.exists(os.path.join(data_path_write, path_write2)):
        	os.makedirs(os.path.join(data_path_write, path_write2))
        for drawer in os.listdir(charactor_path):
            drawer_path = os.path.join(charactor_path, drawer)
            shutil.copyfile(drawer_path, os.path.join(data_path_write, path_write2, drawer))
