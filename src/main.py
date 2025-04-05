from data_downloader import ECGDatasetDownloader
from ecg_preprocessor import ECGPreprocessor
from ecg_segmenter import ECGSegmenter
from model_trainer import ECGModelTrainer
from ecg_analyzer import ECGArrhythmiaAnalyzer
from visualizer import ECGVisualizer

# Example workflow
if __name__ == "__main__":
    # 1. Download data
    downloader = ECGDatasetDownloader()
    downloader.download_dataset()
    
    # 2. Preprocess data
    preprocessor = ECGPreprocessor("mitdb")
    preprocessed = preprocessor.preprocess_all("mitdb_preprocessed")
    
    # 3. Segment data
    segmenter = ECGSegmenter("mitdb", "mitdb_preprocessed")
    segments, labels = segmenter.segment_records(save_dir="mitdb_segmented")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = segmenter.split_dataset(
        segments, labels, save_dir="mitdb_splits"
    )
    
    # 4. Train model
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]
    trainer = ECGModelTrainer(input_shape, num_classes)
    history = trainer.train(X_train, y_train, X_val, y_val)
    trainer.save_model("ecg_model.h5")
    
    # 5. Analyze new data
    analyzer = ECGArrhythmiaAnalyzer("ecg_model.h5")
    result = analyzer.analyze_record("mitdb/100")
    
    # 6. Visualize results
    ECGVisualizer.plot_training_history(history)
    # ... other visualizations as needed