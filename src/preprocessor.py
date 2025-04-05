import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
import os

class ECGPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def bandpass_filter(self, signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def preprocess_record(self, record_name, lowcut=0.5, highcut=50.0):
        record = wfdb.rdrecord(os.path.join(self.data_dir, record_name))
        ecg_signal = record.p_signal[:, 0]
        
        # Normalize
        normalized = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        
        # Filter
        filtered = self.bandpass_filter(normalized, lowcut, highcut, record.fs)
        
        return {
            'original': ecg_signal,
            'normalized': normalized,
            'filtered': filtered,
            'fs': record.fs
        }
    
    def preprocess_all(self, save_dir=None):
        record_files = [f.split('.')[0] for f in os.listdir(self.data_dir) if f.endswith('.dat')]
        preprocessed = {}
        
        for record in record_files:
            preprocessed[record] = self.preprocess_record(record)
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for record_name, data in preprocessed.items():
                np.save(os.path.join(save_dir, f"{record_name}.npy"), data['filtered'])
                
        return preprocessed