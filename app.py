# app.py

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from main import HandwritingDetector # Impor class Anda

app = Flask(__name__)

# --- Inisialisasi Model dan Variabel Global ---
print("Memuat model, harap tunggu...")
detector = HandwritingDetector()
# Pastikan file model .h5 ada di folder yang sama
if not detector.load_model('handwriting_model_augmented.h5'):
    print("===================================================")
    print("PERINGATAN: Model 'handwriting_model_augmented.h5' tidak ditemukan.")
    print("Aplikasi akan berjalan, tapi prediksi tidak akan berfungsi.")
    print("===================================================")

camera = None
use_ensemble = True
last_best_preview = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(last_best_preview, "...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def generate_camera_frames():
    """Generator untuk stream frame kamera utama dengan deteksi."""
    global last_best_preview, use_ensemble

    while True:
        if camera is None or not camera.isOpened():
            # Tampilkan gambar 'kamera mati' jika kamera tidak aktif
            frame_off = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame_off, "Kamera Mati", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_off, "Tekan 'f' untuk menyalakan", (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame_off)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        success, frame = camera.read()
        if not success:
            break
        
        # Logika deteksi dari class Anda
        
        # 1. Tentukan area pendeteksian (60% dari lebar, di tengah)
        full_h, full_w, _ = frame.shape
        detection_ratio = 0.60
        
        detection_w = int(full_w * detection_ratio)
        detection_h = int(full_h * detection_ratio) 
        
        offset_x = int((full_w - detection_w) / 2)
        offset_y = int((full_h - detection_h) / 2)

        # 2. Gambar kotak untuk menandai area deteksi pada frame tampilan
        # Ini untuk visualisasi pengguna
        cv2.rectangle(frame, (offset_x, offset_y), (offset_x + detection_w, offset_y + detection_h), (255, 0, 0), 2)
        cv2.putText(frame, "Area Deteksi", (offset_x + 10, offset_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 3. Potong (crop) frame untuk diproses oleh model
        # Slicing sekarang menggunakan offset_x dan offset_y
        detection_area_frame = frame[offset_y:offset_y + detection_h, offset_x:offset_x + detection_w]

        # 4. Lakukan deteksi HANYA pada area yang sudah dipotong
        contours, thresh, frame_processed_for_roi = detector.detect_contours(detection_area_frame)
        
        best_roi = None
        best_digit = None
        best_confidence = 0.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame_processed_for_roi[y:y+h, x:x+w]
            
            if roi.size > 0:
                digit, confidence = detector.predict_digit(roi, use_ensemble=use_ensemble)
                if digit is not None and confidence > 0.25:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_digit = digit
                        best_roi = roi.copy()
                    
                    # 5. Saat menggambar hasil, tambahkan kembali 'offset_x' DAN 'offset_y'
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                    
                    # Gambar kotak pembatas pada frame penuh
                    cv2.rectangle(frame, (x + offset_x, y + offset_y), (x + offset_x + w, y + offset_y + h), color, 2)
                    
                    # Gambar label pada frame penuh
                    label = f"{digit} ({confidence:.2f})"
                    cv2.putText(frame, label, (x + offset_x, y + offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if best_roi is not None:
            last_best_preview = detector.create_mini_preview(best_roi, best_digit, best_confidence)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_prediction_frames():
    """Generator untuk stream frame mini window prediksi."""
    global last_best_preview
    while True:
        ret, buffer = cv2.imencode('.jpg', last_best_preview)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Menampilkan halaman utama web."""
    # Kirim status awal mode ensemble ke template agar bisa ditampilkan saat load
    initial_mode = 'ON' if use_ensemble else 'OFF'
    return render_template('index.html', initial_ensemble_mode=initial_mode)

@app.route('/video_feed')
def video_feed():
    """Route untuk streaming video kamera utama."""
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction_feed')
def prediction_feed():
    """Route untuk streaming mini window prediksi."""
    return Response(generate_prediction_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Routes untuk Kontrol Keyboard ---

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera
    status = ''
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        status = 'dinyalakan'
    else:
        camera.release()
        camera = None
        status = 'dimatikan'
    return jsonify({'status': f'Kamera {status}'})

@app.route('/toggle_ensemble', methods=['POST'])
def toggle_ensemble():
    global use_ensemble
    use_ensemble = not use_ensemble
    status = 'ON' if use_ensemble else 'OFF'
    # Tambahkan key 'ensemble_mode' untuk mempermudah parsing di JavaScript
    return jsonify({'status': f'Ensemble Mode: {status}', 'ensemble_mode': status})

@app.route('/reset_detection', methods=['POST'])
def reset_detection():
    global last_best_preview
    # Buat ulang tampilan 'No Detection'
    last_best_preview = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.putText(last_best_preview, "Reset", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return jsonify({'status': 'Deteksi direset'})


if __name__ == '__main__':
    # host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan yang sama
    app.run(host='0.0.0.0', debug=True)