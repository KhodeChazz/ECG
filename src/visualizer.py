import matplotlib.pyplot as plt
import numpy as np

class ECGVisualizer:
    @staticmethod
    def plot_ecg_with_annotations(signal, annotations=None, title="ECG Signal"):
        plt.figure(figsize=(15, 5))
        plt.plot(signal, label='ECG Signal', color='blue')
        
        if annotations:
            for idx, symbol in zip(annotations['indices'], annotations['symbols']):
                plt.scatter(idx, signal[idx], color='red')
                plt.text(idx + 35, signal[idx], symbol, color='red', fontsize=8, ha='left')
        
        plt.title(title)
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_processing_steps(raw, normalized, filtered):
        plt.figure(figsize=(15, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(raw, label="Raw Signal", color='blue')
        plt.title("Raw ECG Signal")
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(normalized, label="Normalized Signal", color='green')
        plt.title("Normalized ECG Signal")
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(filtered, label="Filtered Signal", color='red')
        plt.title("Filtered ECG Signal (Noise Reduced)")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_ecg_with_r_peaks_and_st(signal, r_peaks, st_segments, st_abnormalities, sampling_rate=360):
        plt.figure(figsize=(15, 6))
        plt.plot(signal, label='ECG Signal', color='blue', alpha=0.7)
        plt.scatter(r_peaks, signal[r_peaks], color='red', label='R-peaks', zorder=5)
        
        st_duration = int(0.08 * sampling_rate)
        for i, peak in enumerate(r_peaks):
            if peak + st_duration < len(signal):
                st_indices = range(peak, peak + st_duration)
                st_values = signal[st_indices]
                color = 'orange' if st_abnormalities[i] else 'green'
                label = 'Abnormal ST' if st_abnormalities[i] and i == 0 else 'Normal ST' if i == 0 else ""
                plt.plot(st_indices, st_values, color=color, linewidth=2, label=label)
        
        plt.title('ECG with R-peaks and ST-segments')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_training_history(history):
        history_dict = history.history
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='Train Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'], label='Train Loss')
        plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()