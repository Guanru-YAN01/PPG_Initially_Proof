# main.py
import os
from segmentation import process_subject_pkl

data_dir = './dalias'
output_dir = './segmented_data'

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.pkl'):
        fpath = os.path.join(data_dir, fname)
        process_subject_pkl(fpath, output_dir)
