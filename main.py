import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class HandwritingDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def create_model(self):
        """Membuat model CNN untuk klasifikasi digit"""
        model = keras.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train_model(self):
        """Training model menggunakan dataset MNIST"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalisasi data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        print("Creating and training model...")
        self.model = self.create_model()
        
        # Training
        history = self.model.fit(x_train, y_train,
                                epochs=10,
                                batch_size=128,
                                validation_data=(x_test, y_test),
                                verbose=1)
        
        # Evaluasi
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        self.is_trained = True
        return history
    
    def save_model(self, filepath='handwriting_model.h5'):
        """Simpan model yang sudah ditraining"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='handwriting_model.h5'):
        """Load model yang sudah ditraining"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def preprocess_image(self, img):
        """Preprocessing gambar untuk prediksi"""
        # Convert ke grayscale jika perlu
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize ke 28x28
        img = cv2.resize(img, (28, 28))
        
        # Normalisasi
        img = img.astype('float32') / 255.0
        
        # Reshape untuk model
        img = img.reshape(1, 28, 28)
        
        return img
    
    def predict_digit(self, img):
        """Prediksi digit dari gambar"""
        if not self.is_trained:
            return None, 0.0
        
        processed_img = self.preprocess_image(img)
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_digit, confidence
    
    def detect_contours(self, frame):
        """Deteksi kontur tulisan tangan pada frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours berdasarkan area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Filter area yang sesuai
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 2.0:  # Filter aspect ratio
                    valid_contours.append(contour)
        
        return valid_contours, thresh
    
    def run_camera_detection(self):
        """Menjalankan deteksi real-time menggunakan kamera"""
        if not self.is_trained:
            print("Model belum ditraining atau diload!")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        print("Deteksi handwriting dimulai. Tekan 'q' untuk keluar.")
        print("Instruksi: Tulis digit (0-9) di depan kamera dengan tinta hitam di kertas putih")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame untuk pengalaman yang lebih natural
            frame = cv2.flip(frame, 1)
            
            # Deteksi kontur
            contours, thresh = self.detect_contours(frame)
            
            # Process setiap kontur yang valid
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                if roi.size > 0:
                    # Prediksi digit
                    digit, confidence = self.predict_digit(roi)
                    
                    if digit is not None and confidence > 0.5:  # Threshold confidence
                        # Gambar bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Tampilkan hasil prediksi
                        label = f"Digit: {digit} ({confidence:.2f})"
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Tampilkan instruksi
            cv2.putText(frame, "Tulis digit 0-9 dengan tinta hitam", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Tekan 'q' untuk keluar", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Tampilkan frame
            cv2.imshow('Handwriting Detection', frame)
            cv2.imshow('Threshold', thresh)
            
            # Break jika 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = HandwritingDetector()
    
    # Coba load model yang sudah ada
    if not detector.load_model():
        print("Model tidak ditemukan. Mulai training...")
        detector.train_model()
        detector.save_model()
    
    # Jalankan deteksi kamera
    detector.run_camera_detection()

if __name__ == "__main__":
    main()
