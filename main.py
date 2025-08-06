import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


class HandwritingDetector:
    def __init__(self):
        """Inisialisasi HandwritingDetector."""
        self.model = None
        self.is_trained = False

    def create_model(self):
        """Membuat model CNN untuk klasifikasi digit."""
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

    def create_data_augmentation_generator(self):
        """Membuat data augmentation generator untuk training."""
        datagen = ImageDataGenerator(
            rotation_range=10,        # Rotasi ±10 derajat
            width_shift_range=0.1,    # Shift horizontal 10%
            height_shift_range=0.1,   # Shift vertikal 10%
            shear_range=0.1,          # Shear transformation
            zoom_range=0.1,           # Zoom in/out 10%
            horizontal_flip=False,    # Tidak flip horizontal untuk digit
            fill_mode='constant',     # Isi area kosong dengan 0 (hitam)
            cval=0.0
        )
        return datagen

    def train_model(self):
        """Training model menggunakan dataset MNIST dengan data augmentation."""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Normalisasi data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape untuk data augmentation (tambah channel dimension)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        print("Creating and training model with data augmentation...")
        self.model = self.create_model()

        # Buat data augmentation generator
        datagen = self.create_data_augmentation_generator()
        datagen.fit(x_train)

        # Training dengan data augmentation
        print("Training dengan data augmentation...")
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            steps_per_epoch=len(x_train) // 128,
            epochs=15,  # Lebih banyak epochs karena data augmentation
            validation_data=(x_test, y_test),
            verbose=1
        )

        # Evaluasi
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")

        self.is_trained = True
        return history

    def save_model(self, filepath='handwriting_model_augmented.h5'):
        """Simpan model yang sudah ditraining."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath='handwriting_model_augmented.h5'):
        """Load model yang sudah ditraining."""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        return False

    def preprocess_image(self, img, apply_augmentation=False):
        """Preprocessing gambar untuk prediksi dengan opsi augmentation."""
        # Convert ke grayscale jika perlu
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mirror horizontal untuk konsistensi dengan training data
        #img = cv2.flip(img, 1)

        # Resize ke 28x28
        img = cv2.resize(img, (28, 28))

        # Additional preprocessing untuk meningkatkan akurasi
        # 1. Gaussian blur untuk mengurangi noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 2. Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # 3. Contrast enhancement menggunakan CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

        # 4. Threshold adaptif untuk better binarization
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Normalisasi
        img = img.astype('float32') / 255.0

        # Jika apply_augmentation True, lakukan augmentasi ringan
        if apply_augmentation:
            img = self.apply_test_time_augmentation(img)

        # Reshape untuk model
        img = img.reshape(1, 28, 28, 1)

        return img

    def apply_test_time_augmentation(self, img):
        """Aplikasi augmentasi ringan saat testing untuk meningkatkan robustness."""
        # Rotasi kecil secara random
        angle = np.random.uniform(-5, 5)
        center = (14, 14)  # Center of 28x28 image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (28, 28))

        # Slight translation
        tx = np.random.uniform(-2, 2)
        ty = np.random.uniform(-2, 2)
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        img = cv2.warpAffine(img, translation_matrix, (28, 28))

        return img

    def predict_digit_with_ensemble(self, img, n_predictions=5):
        """Prediksi dengan ensemble untuk meningkatkan akurasi."""
        if not self.is_trained:
            return None, 0.0

        predictions = []

        # Lakukan beberapa prediksi dengan augmentasi berbeda
        for i in range(n_predictions):
            processed_img = self.preprocess_image(img, apply_augmentation=True)
            pred = self.model.predict(processed_img, verbose=0)
            predictions.append(pred[0])

        # Rata-rata dari semua prediksi
        avg_predictions = np.mean(predictions, axis=0)
        predicted_digit = np.argmax(avg_predictions)
        confidence = np.max(avg_predictions)

        return predicted_digit, confidence

    def predict_digit(self, img, use_ensemble=True):
        """Prediksi digit dari gambar dengan opsi ensemble."""
        if use_ensemble:
            return self.predict_digit_with_ensemble(img, n_predictions=3)
        else:
            if not self.is_trained:
                return None, 0.0

            processed_img = self.preprocess_image(img, apply_augmentation=False)
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            return predicted_digit, confidence

    def detect_contours(self, frame):
        """Deteksi kontur tulisan tangan pada frame dengan preprocessing yang lebih baik."""
        # Mirror frame horizontal untuk natural writing experience
        #frame_mirrored = cv2.flip(frame, 1)

        #gray = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhanced preprocessing pipeline
        # 1. Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. CLAHE untuk contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # 3. Adaptive threshold untuk better binarization
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

        # 4. Morphological operations untuk cleanup
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Enhanced contour filtering
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 15000:  # Expanded area range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # More flexible aspect ratio and size constraints
                if 0.15 < aspect_ratio < 3.0 and w > 20 and h > 20:
                    # Check contour solidity (untuk filter noise)
                    hull = cv2.convexHull(contour)
                    solidity = float(area) / cv2.contourArea(hull)

                    if solidity > 0.3:  # Filter out very irregular shapes
                        valid_contours.append(contour)

        return valid_contours, thresh, frame #frame_mirrored

    def create_mini_preview(self, roi, digit, confidence, size=(150, 150)):
        """Membuat mini preview dari ROI yang terdeteksi."""
        if roi.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Resize ROI untuk preview
        roi_resized = cv2.resize(roi, size)

        # Konversi ke BGR jika grayscale
        if len(roi_resized.shape) == 2:
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

        # Tambahkan border hijau jika confident
        if confidence > 0.5:
            cv2.rectangle(roi_resized, (0, 0), (size[0] - 1, size[1] - 1), (0, 255, 0), 3)
        else:
            cv2.rectangle(roi_resized, (0, 0), (size[0] - 1, size[1] - 1), (0, 0, 255), 3)

        # Tambahkan text prediksi
        text = f"{digit}" if digit is not None else "?"
        cv2.putText(roi_resized, text, (size[0] // 2 - 20, size[1] // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        return roi_resized

    def run_camera_detection(self):
        """Menjalankan deteksi real-time menggunakan kamera."""
        if not self.is_trained:
            print("Model belum ditraining atau diload!")
            return

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return

        print("Deteksi handwriting dimulai. Tekan 'q' untuk keluar.")
        print("Instruksi: Tulis digit (0-9) di depan kamera dengan tinta hitam di kertas putih")
        print("Kontrol:")
        print("  'q' - Keluar")
        print("  'r' - Reset detection")
        print("  'e' - Toggle ensemble mode")
        print("  's' - Save current detection")

        # Atur posisi window
        cv2.namedWindow('Handwriting Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Threshold View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection Preview', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Raw Camera Feed', cv2.WINDOW_NORMAL)

        # Posisikan window
        cv2.moveWindow('Handwriting Detection', 100, 100)
        cv2.moveWindow('Threshold View', 700, 100)
        cv2.moveWindow('Detection Preview', 100, 500)
        cv2.moveWindow('Raw Camera Feed', 700, 500)

        # Resize window
        cv2.resizeWindow('Detection Preview', 200, 200)
        cv2.resizeWindow('Raw Camera Feed', 300, 240)

        # Settings
        use_ensemble = True
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Simpan frame asli untuk raw camera feed (tidak di-mirror)
            raw_frame = frame.copy()

            # Deteksi kontur (sudah include mirroring di dalam fungsi)
            contours, thresh, frame_processed = self.detect_contours(frame)

            # Variable untuk menyimpan ROI terbaik
            best_roi = None
            best_digit = None
            best_confidence = 0.0
            detection_count = 0

            # Process setiap kontur yang valid
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Extract ROI dari frame yang sudah di-mirror
                roi = frame_processed[y:y + h, x:x + w]

                if roi.size > 0:
                    # Prediksi digit dengan atau tanpa ensemble
                    digit, confidence = self.predict_digit(roi, use_ensemble=use_ensemble)

                    if digit is not None and confidence > 0.25:  # Threshold confidence lebih rendah
                        detection_count += 1

                        # Simpan ROI dengan confidence tertinggi
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_digit = digit
                            best_roi = roi.copy()

                        # Gambar bounding box dengan warna berdasarkan confidence
                        if confidence > 0.7:
                            color = (0, 255, 0)      # Hijau - Very confident
                        elif confidence > 0.5:
                            color = (0, 255, 255)    # Kuning - Confident
                        else:
                            color = (0, 165, 255)    # Orange - Less confident

                        cv2.rectangle(frame_processed, (x, y), (x + w, y + h), color, 2)

                        # Tampilkan hasil prediksi
                        label = f"Digit: {digit} ({confidence:.3f})"
                        cv2.putText(frame_processed, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tampilkan informasi dan instruksi
            mode_text = "Ensemble ON" if use_ensemble else "Ensemble OFF"
            cv2.putText(frame_processed, "Tulis digit 0-9 (mirrored input)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_processed, f"Mode: {mode_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_processed, f"Deteksi: {detection_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_processed, f"Saved: {saved_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Control instructions
            cv2.putText(frame_processed, "Controls: q=quit, r=reset, e=ensemble, s=save", (10, frame_processed.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Buat mini preview dari deteksi terbaik
            if best_roi is not None:
                mini_preview = self.create_mini_preview(best_roi, best_digit, best_confidence)
            else:
                # Buat preview kosong jika tidak ada deteksi
                mini_preview = np.zeros((150, 150, 3), dtype=np.uint8)
                cv2.putText(mini_preview, "No Detection", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Resize raw camera feed untuk mini window
            raw_mini = cv2.resize(raw_frame, (300, 240))

            # Tampilkan semua window
            cv2.imshow('Handwriting Detection', frame_processed)
            cv2.imshow('Threshold View', thresh)
            cv2.imshow('Detection Preview', mini_preview)
            cv2.imshow('Raw Camera Feed', raw_mini)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset detection
                best_roi = None
                best_digit = None
                best_confidence = 0.0
                print("Detection reset")
            elif key == ord('e'):  # Toggle ensemble mode
                use_ensemble = not use_ensemble
                print(f"Ensemble mode: {'ON' if use_ensemble else 'OFF'}")
            elif key == ord('s') and best_roi is not None:  # Save current detection
                filename = f"detection_{saved_count:03d}_digit_{best_digit}.png"
                cv2.imwrite(filename, best_roi)
                saved_count += 1
                print(f"Saved detection as {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total detections saved: {saved_count}")


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