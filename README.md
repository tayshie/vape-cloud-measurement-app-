# Vape Cloud Measurement App

A desktop application that measures the relative size of vape clouds in real-time using a webcam feed. The application uses OpenCV for video processing and PyQt5 for the graphical user interface.

## Features

- **Live Webcam Feed:** Displays live video from your camera.
- **Real-Time Vape Cloud Detection:** Detects vape clouds and draws a bounding box on the video.
- **Cloud Size Measurement:** Measures the detected cloud's size in pixel area.
- **Data Logging:** Optionally record measurements with timestamps and export to CSV.
- **Adjustable Sensitivity:** Use the slider to adjust detection sensitivity based on lighting conditions.

## Prerequisites

- Python 3.7 or later.
- A working webcam.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/vape-cloud-measurement-app.git
   cd vape-cloud-measurement-app

(Optional) Create a Virtual Environment:

python -m venv venv
# Activate the virtual environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install Dependencies:

pip install opencv-python PyQt5
Running the App
Run the following command from your project directory:

python vape_cloud_app.py
This will launch the application window where you can start measuring vape cloud sizes.

Usage
Start Measurement: Click the "Start Measurement" button.
Stop: Click the "Stop" button to stop the video feed.
Camera Selector: Use the dropdown to choose between available cameras.
Adjust Sensitivity: Move the slider to adjust detection sensitivity.
Record Data and Export: Check "Record Data" to log measurements and click "Export CSV" to save the data.
Troubleshooting
No Webcam Feed: Ensure your webcam is connected and not used by another application.
Adjustment Issues: Modify the sensitivity slider or tweak the HSV thresholds in vape_cloud_app.py if needed.
License
MIT License