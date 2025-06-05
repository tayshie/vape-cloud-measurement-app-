#!/usr/bin/env python3
"""
Vape Cloud Measurement App
---------------------------
A desktop application that uses a webcam feed to detect vape clouds in real-time,
highlight them with a bounding box, and measure the (relative) size of the detected cloud.
The GUI is developed using PyQt5 while OpenCV handles video capture and image processing.

This version:
  - Scans for camera devices indices 0-9.
  - Displays a live feed from the selected device.
  - Works with your OBS Virtual Camera (make sure it’s running in OBS before you start).
"""

import os
# Set the environment variable to suppress extra OpenCV log messages.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import sys
import cv2
import csv
import numpy as np
from datetime import datetime
from contextlib import redirect_stderr

# PyQt5 imports:
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QCheckBox, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont

# ------------------------------------------------------------------------------------
# Video Capture Thread
# ------------------------------------------------------------------------------------
class VideoThread(QThread):
    # Signal to transmit the processed video frame.
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # Signal to transmit the vape cloud measurement (area in pixels).
    measurement_signal = pyqtSignal(float)

    def __init__(self, cam_index=0, sensitivity=50):
        super().__init__()
        self._run_flag = True
        self.camera_index = cam_index
        self.sensitivity = sensitivity  # Sensitivity for time-of-day adjustments in detection
    
    def run(self):
        # Open the selected camera device using the default backend.
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open video device with index {self.camera_index}.")
            self._run_flag = False

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process the captured frame, detect vape cloud, and measure its (relative) size.
                processed_frame, measurement = detect_vape_cloud(frame, self.sensitivity)
                self.change_pixmap_signal.emit(processed_frame)
                self.measurement_signal.emit(measurement)
            else:
                print("Failed to grab frame")
            cv2.waitKey(1)  # Short delay to reduce CPU load
        cap.release()

    def stop(self):
        """Stops the video capture thread."""
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def update_camera_index(self, cam_index):
        self.camera_index = cam_index

# ------------------------------------------------------------------------------------
# Vape Cloud Detection Function
# ------------------------------------------------------------------------------------
def detect_vape_cloud(frame, sensitivity=50):
    """
    Process the frame to detect a vape cloud:
      - Converts the frame to HSV color space.
      - Thresholds to isolate bright regions (vape cloud–like areas).
      - Removes noise using morphological opening.
      - Finds contours, selects the largest, and draws a bounding box.
      - Returns the processed frame and the cloud's area (in pixels).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, sensitivity])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measurement = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        measurement = w * h    # Relative area in pixel units
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {measurement:,} px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, measurement

# ------------------------------------------------------------------------------------
# Main Application Window (GUI)
# ------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vape Cloud Measurement App")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2e2e2e; color: #FFF;")
        
        # Central container widget.
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        
        # Video display area.
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000;")
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        
        # Label to display live measurement (area) in pixels.
        self.measurement_label = QLabel("Cloud Size: 0 px")
        self.measurement_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.measurement_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.measurement_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Measurement")
        self.start_button.clicked.connect(self.start_video)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video)
        controls_layout.addWidget(self.stop_button)
        
        # Camera selector drop-down.
        self.cam_selector = QComboBox()
        self.populate_camera_list()  # Scan indices 0-9
        controls_layout.addWidget(self.cam_selector)
        
        # Sensitivity slider.
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(255)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setToolTip("Adjust detection sensitivity")
        self.sensitivity_slider.valueChanged.connect(self.sensitivity_changed)
        controls_layout.addWidget(QLabel("Sensitivity:"))
        controls_layout.addWidget(self.sensitivity_slider)
        
        # Data logging and export options.
        self.log_checkbox = QCheckBox("Record Data")
        controls_layout.addWidget(self.log_checkbox)
        
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        controls_layout.addWidget(self.export_button)
        
        main_layout.addLayout(controls_layout)
        
        self.measurement_data = []  # List to store (timestamp, measurement) tuples.
        self.thread = None

    def populate_camera_list(self):
        """
        Scans for available camera devices by checking indices 0–9.
        If the OBS Virtual Camera is running, it should appear in the list.
        (Select the desired device from the drop-down.)
        """
        self.cam_selector.clear()
        available_indices = []
        MAX_CAMERAS_TO_CHECK = 10
        for i in range(MAX_CAMERAS_TO_CHECK):
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        available_indices.append(i)
                    cap.release()
        if not available_indices:
            print("No cameras detected. Defaulting to index 0.")
            available_indices = [0]
        # Add each detected camera to the selector.
        for idx in available_indices:
            self.cam_selector.addItem(f"Camera {idx}", idx)

    def start_video(self):
        # Get the selected camera index and sensitivity.
        cam_index = self.cam_selector.currentData()
        sensitivity = self.sensitivity_slider.value()
        self.thread = VideoThread(cam_index=cam_index, sensitivity=sensitivity)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.measurement_signal.connect(self.update_measurement)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cam_selector.setEnabled(False)
        if self.log_checkbox.isChecked():
            self.measurement_data = []

    def stop_video(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cam_selector.setEnabled(True)

    def update_image(self, cv_img):
        """Converts the captured frame from BGR to RGB and displays it."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def update_measurement(self, value):
        self.measurement_label.setText(f"Cloud Size: {value:,} px")
        if self.log_checkbox.isChecked():
            timestamp = datetime.now().isoformat(timespec="seconds")
            self.measurement_data.append((timestamp, value))

    def sensitivity_changed(self, value):
        if self.thread is not None:
            self.thread.update_sensitivity(value)

    def export_csv(self):
        if not self.measurement_data:
            print("No measurement data to export.")
            return
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Measurement Data", "", "CSV Files (*.csv)", options=options
        )
        if filename:
            try:
                with open(filename, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(["Timestamp", "Cloud Size (px)"])
                    csvwriter.writerows(self.measurement_data)
                print(f"Data exported successfully to {filename}")
            except Exception as e:
                print("Failed to export data:", e)

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
        event.accept()

# ------------------------------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())