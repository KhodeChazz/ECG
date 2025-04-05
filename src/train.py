from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class ECGModelTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        model = models.Sequential()
        
        model.add(layers.Conv1D(32, 5, activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling1D(2))
        
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(64))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
        )
        return history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, filepath):
        self.model.save(filepath)