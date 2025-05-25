import os
import pickle
import numpy as np
from preprocessing import detrend_signal, remove_baseline, butterworth_lowpass, normalize_minmax, normalize_envelope

def process_file(input_path: str,
                 output_path: str,
                 fs: float = 64.0,
                 baseline_win: int = 64,
                 cutoff: float = 6.0,
                 use_envelope: bool = True):
    
    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    preprocessed = []
    for sample in samples: 
        bvp = np.asarray(sample['bvp']).squeeze()                 
        label = sample.get('label', None)      
        activity_id = sample.get('activity_id')
        activity_name = sample.get('activity_name')

        # preprocessing
        x = detrend_signal(bvp)                               
        x = remove_baseline(x, window_size=baseline_win)       
        x = butterworth_lowpass(x, cutoff=cutoff, fs=fs)     
        x = normalize_envelope(x) if use_envelope else x == normalize_minmax(x)                       

        # save
        preprocessed.append({
            'bvp_preprocessed': x,
            'label': label,
            'activity_id': activity_id,
            'activity_name': activity_name
                            })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed, f)


def main(input_root: str = 'segmented_data',
         output_root: str = 'preprocessed_data',
         **kwargs):
    for subj in sorted(os.listdir(input_root)):
        subj_in = os.path.join(input_root, subj)
        subj_out = os.path.join(output_root, subj)
        if not os.path.isdir(subj_in):
            continue

        for category in os.listdir(subj_in):
            cat_in = os.path.join(subj_in, category)
            cat_out = os.path.join(subj_out, category)
            if not os.path.isdir(cat_in):
                continue

            for fname in os.listdir(cat_in):
                if not fname.endswith('.pkl'):
                    continue
                in_path = os.path.join(cat_in, fname)
                out_path = os.path.join(cat_out, fname)
                process_file(in_path, out_path, **kwargs)


if __name__ == '__main__':
    main()