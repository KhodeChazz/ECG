import numpy as np
import wfdb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class ECGSegmenter:
    def __init__(self, data_dir, preprocessed_dir):
        self.data_dir = data_dir
        self.preprocessed_dir = preprocessed_dir
        
    def segment_records(self, window_size=360, save_dir=None):
        record_files = [f.split('.')[0] for f in os.listdir(self.preprocessed_dir) if f.endswith('.npy')]
        all_segments = []
        all_labels = []
        
        for record_name in record_files:
            signal = np.load(os.path.join(self.preprocessed_dir, f"{record_name}.npy"))
            annotation = wfdb.rdann(os.path.join(self.data_dir, record_name), 'atr')
            
            segments, labels = self._segment_signal(signal, annotation, window_size)
            all_segments.extend(segments)
            all_labels.extend(labels)
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "all_segments.npy"), np.array(all_segments))
            np.save(os.path.join(save_dir, "all_labels.npy"), np.array(all_labels))
            
        return np.array(all_segments), np.array(all_labels)
    
    def _segment_signal(self, signal, annotation, window_size):
        segments = []
        labels = []
        
        for idx, sample in enumerate(annotation.sample):
            if sample - window_size >= 0 and sample + window_size < len(signal):
                segment = signal[sample - window_size : sample + window_size]
                segments.append(segment)
                labels.append(annotation.symbol[idx])
                
        return segments, labels
    
    def split_dataset(self, segments, labels, test_size=0.3, val_size=0.5, save_dir=None):
        # First split into train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            segments, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Then split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        # One-hot encode labels
        label_encoder = LabelEncoder()
        y_train_enc = to_categorical(label_encoder.fit_transform(y_train))
        y_val_enc = to_categorical(label_encoder.transform(y_val))
        y_test_enc = to_categorical(label_encoder.transform(y_test))
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "X_train.npy"), X_train)
            np.save(os.path.join(save_dir, "X_val.npy"), X_val)
            np.save(os.path.join(save_dir, "X_test.npy"), X_test)
            np.save(os.path.join(save_dir, "y_train.npy"), y_train_enc)
            np.save(os.path.join(save_dir, "y_val.npy"), y_val_enc)
            np.save(os.path.join(save_dir, "y_test.npy"), y_test_enc)
            
        return (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc), label_encoder
