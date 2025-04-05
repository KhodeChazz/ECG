import numpy as np
import wfdb
import os
from scipy.signal import find_peaks, savgol_filter
from tensorflow.keras.models import load_model

class ECGArrhythmiaAnalyzer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.st_elevation_thresh = 0.2  # 2 mm
        self.st_depression_thresh = -0.1  # -1 mm
        self.persistence_threshold = 8
        
    def analyze_record(self, record_path):
        # Load and preprocess signal
        signal, _ = wfdb.rdsamp(record_path)
        ecg_signal = signal[:, 0]
        ecg_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        
        # Detect R-peaks
        r_peaks, _ = find_peaks(ecg_signal, distance=360 // 2)
        
        # Arrhythmia detection
        arrhythmia_result = self._detect_arrhythmias(ecg_signal)
        
        # ST-segment analysis
        st_result = self._analyze_st_segments(ecg_signal, r_peaks)
        
        return {
            'arrhythmia': arrhythmia_result,
            'st_abnormalities': st_result,
            'r_peaks': r_peaks
        }
    
    def _detect_arrhythmias(self, ecg_signal, segment_length=360):
        segments = []
        for i in range(0, len(ecg_signal) - segment_length, segment_length):
            segments.append(ecg_signal[i:i + segment_length])
        segments = np.array(segments).reshape(-1, segment_length, 1)
        predictions = self.model.predict(segments)
        return np.argmax(predictions, axis=1)
    
    def _analyze_st_segments(self, ecg_signal, r_peaks):
        st_segments = self._calculate_st_segments(ecg_signal, r_peaks)
        smoothed = self._smooth_st_segments(st_segments)
        abnormalities = self._detect_abnormal_st(smoothed)
        return self._check_persistence(abnormalities)
    
    def _calculate_st_segments(self, ecg_signal, r_peaks, sampling_rate=360):
        st_segments = []
        st_duration = int(0.08 * sampling_rate)
        for peak in r_peaks:
            if peak + st_duration < len(ecg_signal):
                st_segment = ecg_signal[peak:peak + st_duration]
                st_segments.append(np.mean(st_segment))
        return np.array(st_segments)
    
    def _smooth_st_segments(self, st_values, window_length=5):
        return savgol_filter(st_values, window_length=window_length, polyorder=2)
    
    def _detect_abnormal_st(self, st_values):
        abnormalities = []
        for st in st_values:
            if st > self.st_elevation_thresh or st < self.st_depression_thresh:
                abnormalities.append(True)
            else:
                abnormalities.append(False)
        return abnormalities
    
    def _check_persistence(self, abnormalities):
        persistent_count = 0
        abnormal_groups = []
        for i in range(1, len(abnormalities)):
            if abnormalities[i] and abnormalities[i - 1]:
                persistent_count += 1
            else:
                if persistent_count >= self.persistence_threshold:
                    abnormal_groups.append(persistent_count)
                persistent_count = 0
        if persistent_count >= self.persistence_threshold:
            abnormal_groups.append(persistent_count)
        return len(abnormal_groups) > 1