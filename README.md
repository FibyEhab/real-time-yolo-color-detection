# Real-Time Object Detection with YOLOv8x and Dynamic Colors

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8x-00FFFF?style=flat&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## About

Advanced real-time object detection system using **YOLOv8x** with **dynamic random color assignment** for each detected class. Each object class receives a unique color that remains **consistent throughout the video stream**, making it easy to track specific object categories.

**Key Features:**
- Real-time detection at 25-30 FPS on CPU
- **Unique random colors per class** (consistent across frames)
- Confidence scores for each detection
- Live webcam stream processing
- YOLOv8x model (highest accuracy variant)
- Easy customization and extension

---

## What Makes This Different?

Unlike standard object detection, this project assigns a **unique color to each detected object class**:
- First detection of a class → **random color assigned**
- All subsequent detections → **same color used**
- Makes tracking specific objects much easier visually

**Example:**
```
Person → Green (consistent)
Car → Blue (consistent)
Dog → Red (consistent)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | YOLOv8x (Ultralytics) |
| **Computer Vision** | OpenCV 4.8+ |
| **Language** | Python 3.8+ |
| **Deep Learning Framework** | PyTorch 2.0+ |
| **Data Processing** | NumPy, Pandas |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam connected to your computer
- 4GB+ RAM (8GB recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/FibyEhab/real-time-yolo-color-detection.git
cd real-time-yolo-color-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**First run note:** The YOLOv8x model (~271MB) will be downloaded automatically on first run.

---

## Usage

### Basic Usage
```bash
python main.py
```

### What Happens
1. **Webcam opens** with 1280×480 resolution
2. **YOLOv8x processes** each frame in real-time
3. **Objects are detected** with bounding boxes
4. **Colors are assigned** - first detection of each class gets a random color
5. **Confidence scores** displayed for each detection
6. **Press 'Q'** to quit the program

---

## How the Color System Works

### Color Assignment Logic
```python
# Dictionary stores: class_id → BGR_color
class_colors = {}

# On FIRST detection of a class:
if cls not in class_colors:
    # Generate random BGR color (0-255 for each channel)
    class_colors[cls] = [random.randint(0, 255) for _ in range(3)]

# On ALL subsequent detections:
color = class_colors[cls]  # Use the SAME color
```

### Why This Matters
- **Easy visual tracking** - identify the same object type across frames
- **Professional appearance** - consistent, organized visualization
- **No color conflicts** - each class has its own unique color
- **Memory efficient** - only stores one color per class

---

## Customization Guide

### Option 1: Manual Color Assignment
Instead of random colors, assign specific colors to specific classes:

```python
# Define colors manually (BGR format)
class_colors = {
    0: [0, 255, 0],      # Person → Green
    2: [255, 0, 0],      # Car → Blue (BGR)
    5: [0, 0, 255],      # Dog → Red
    # Add more as needed
}
```

### Option 2: Reproducible Colors (Fixed Seed)
Get the same colors every run:

```python
random.seed(42)  # Add before the while loop
```

### Option 3: Adjust Confidence Threshold
Higher values = fewer false positives:

```python
results = model.predict(frame, conf=0.7)  # Default: 0.5
```

### Option 4: Change Resolution
For faster processing or better detail:

```python
cap.set(3, 1024)  # Width
cap.set(4, 384)   # Height
```

### Option 5: Use Different YOLOv8 Variants
```python
# Speed vs Accuracy tradeoff:
model = YOLO("yolov8n.pt")  # Nano (fastest)
model = YOLO("yolov8s.pt")  # Small
model = YOLO("yolov8m.pt")  # Medium RECOMMENDED FOR BALANCE
model = YOLO("yolov8l.pt")  # Large
model = YOLO("yolov8x.pt")  # Extra Large (slowest but most accurate)
```

### Option 6: Process Video File Instead of Webcam
```python
cap = cv2.VideoCapture("path/to/video.mp4")
```

### Option 7: Adjust Visual Properties
```python
# Thicker bounding boxes
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)  # Default: 3

# Larger text
cv2.putText(frame, text, (x1, y1 - 10), 
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)  # Default: 1
```

---

## Performance Metrics

| Aspect | Details |
|--------|---------|
| **Model** | YOLOv8x (Extra Large) |
| **FPS** | 25-30 (CPU) / 60+ (GPU) |
| **Resolution** | 1280×480 (adjustable) |
| **Confidence Threshold** | 0.5 (adjustable) |
| **Detectable Classes** | 80 (COCO Dataset) |
| **Model Size** | 271MB |
| **Memory Usage** | ~2GB RAM |

### FPS by Model Variant (CPU):
- YOLOv8n: 40-50 FPS
- YOLOv8s: 35-40 FPS
- YOLOv8m: 30-35 FPS
- YOLOv8l: 20-25 FPS
- YOLOv8x: 15-30 FPS ← (This project)

---

## Advanced Configuration

### For GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Better Performance on Slow Hardware
```python
# Use YOLOv8m instead and reduce resolution
model = YOLO("yolov8m.pt")
cap.set(3, 640)
cap.set(4, 480)
```

### For Production Deployment
```python
# Add frame skipping (process every nth frame)
frame_count = 0
skip_frames = 2

while True:
    success, frame = cap.read()
    if frame_count % skip_frames == 0:
        results = model.predict(frame, ...)
    frame_count += 1
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
```bash
pip install --upgrade ultralytics opencv-python torch
```

### Issue: Webcam not detected
```bash
python -c "import cv2; print(cv2.VideoCapture(0).get(cv2.CAP_PROP_FRAME_WIDTH))"
```
If output is 0, your webcam isn't detected.

### Issue: FPS too low
1. Use YOLOv8m instead of yolov8x
2. Reduce resolution to 640×480
3. Enable GPU acceleration
4. Close other applications

### Issue: "CUDA out of memory" (GPU users)
```python
model = YOLO("yolov8m.pt")  # Use smaller model
```

### Issue: Camera permission denied (Mac/Linux)
Grant camera access to your terminal:
- **Mac:** System Preferences → Security & Privacy → Camera → Terminal ✓
- **Linux:** `sudo usermod -a -G video $USER`

---

## 📚 Learning Resources

- **[YOLOv8 Official Docs](https://docs.ultralytics.com/)** - Complete documentation
- **[OpenCV Tutorials](https://docs.opencv.org/)** - Computer vision basics
- **[YOLO Research Paper](https://arxiv.org/abs/2004.10934)** - Technical details
- **[COCO Dataset](https://cocodataset.org/)** - 80 detectable classes
- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)** - Deep learning framework

---

## Next Steps / Extensions

Want to enhance this project?

1. **Add Person Counting** - Track and count detected persons
2. **Add Tracking** - Use DeepSORT or SORT to track objects across frames
3. **Add Recording** - Save video with detections
4. **Add Statistics** - Count detections per class per minute
5. **Add GUI** - Create a desktop application with Tkinter/PyQt
6. **Add Web Interface** - Deploy with Flask/FastAPI
7. **Add Multi-threaded Processing** - Process multiple frames simultaneously

---

## Project Structure

```
real-time-yolo-color-detection/
├── main.py                 # Main detection script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── images/
│   └── demo.png           # Demo screenshot
└── .gitignore             # Git ignore rules
```

---

## Contributing

Found a bug? Have an idea? Contributions are welcome!

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**YOLOv8 Model:** Licensed under AGPL-3.0 by Ultralytics

---

## Author

**Fiby Ehab** - AI Engineer

**Email:** your@email.com

**LinkedIn:** [Fiby Ehab](https://www.linkedin.com/in/fiby-ehab-270b55286/)

**GitHub:** [@FibyEhab](https://github.com/FibyEhab)  

Feel free to reach out for questions, suggestions, or collaborations!

---

## Support This Project

If this project was useful to you:
- **⭐ Star** this repository
- **Fork** it for your own use
- **Share** it with others
- **Comment** your feedback

Every star motivates me to keep improving and creating more projects!

---

**Made with ❤️ for the AI community**
