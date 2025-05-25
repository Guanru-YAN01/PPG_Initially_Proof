'''
Data Segmentation Script: Guanru YAN 5/22/2025

This script segments continuous sensor data into separate activities based on labeled events,
and categorizes the resulting segments into predefined motion types:
- minimal (e.g., Sitting, Working)
- rhythmic (e.g., Stairs, Cycling, Walking)
- irregular (e.g., Soccer, Driving, Lunch)

Each segmented activity is saved individually in a structured directory:
  /processed_data/
    └── S1/
        ├── minimal/
        │   ├── Sitting.pkl
        │   └── Working.pkl
        ├── rhythmic/
        │   ├── Stairs.pkl
        │   ├── Cycling.pkl
        │   └── Walking.pkl
        └── irregular/
            ├── Soccer.pkl
            ├── Driving.pkl
            └── Lunch.pkl
'''

import os
import pickle
import numpy as np
from collections import defaultdict

# activity types
activity_info = {
    1: ('Sitting', 'minimal'),
    2: ('Stairs', 'rhythmic'),
    3: ('Soccer', 'irregular'),
    4: ('Cycling', 'rhythmic'),
    5: ('Driving', 'irregular'),
    6: ('Lunch', 'irregular'),
    7: ('Walking', 'rhythmic'),
    8: ('Working', 'minimal')
}

def ensure_dirs(base_path, subject_id):
    '''mkdir for 3 types of activity'''
    for category in ['minimal', 'rhythmic', 'irregular']:
        path = os.path.join(base_path, subject_id, category)
        os.makedirs(path, exist_ok=True)

def extract_windows(signal, window_size, step_size):
    '''sliding win'''
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start:start + window_size])
    return np.array(windows)

def process_subject_pkl(file_path, output_base):
    subject_id = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    bvp = np.asarray(data['signal']['wrist']['BVP'])  # 64 Hz
    acc = np.asarray(data['signal']['wrist']['ACC'])  # 32 Hz
    labels = np.asarray(data['label'])
    activities = np.asarray(data['activity']).flatten()  # 4 Hz

    # sliding win parameters
    bvp_win = 512  # 8s * 64Hz
    acc_win = 256  # 8s * 32Hz
    step = 128     # 2s shift * 64Hz for BVP, implies 64 steps for ACC

    bvp_windows = extract_windows(bvp, bvp_win, step)
    acc_windows = extract_windows(acc, acc_win, step // 2)

    ensure_dirs(output_base, subject_id)

    activity_by_id = defaultdict(list)

    for i, label in enumerate(labels):
        activity_start_idx = i * step // 16  # 1 label/2s, activity 4Hz = 4 values/s
        activity_id = int(np.round(np.mean(activities[activity_start_idx:activity_start_idx+32])))  # 8s
        
        if activity_id not in activity_info:
            continue

        name, category = activity_info[activity_id]
        sample = {
            'bvp': bvp_windows[i],
            'acc': acc_windows[i],
            'label': float(label),
            'activity_id': activity_id,
            'activity_name': name
        }

        activity_by_id[(name, category)].append(sample)

    # save
    for (name, category), samples in activity_by_id.items():
        out_path = os.path.join(output_base, subject_id, category, f"{name}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(samples, f)
