# DETR Vision Project

A personal Computer Vision project for object detection using Facebook's DETR (DEtection TRansformer) model.

## Features

- Object detection on images
- Real-time object detection from webcam
- Clean, modular, and well-documented code
- Command-line interface
- Support for GPU acceleration

## Project Structure

```
detr_vision_project/
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── setup.py                # Package installation script
├── .gitignore              # Files to ignore in version control
├── src/                    # Source code directory
│   └── detr_vision/        # Main package
│       ├── __init__.py     # Makes the directory a package
│       ├── model.py        # DETR model loading and inference
│       ├── camera.py       # Webcam access and processing
│       ├── visualization.py # Result visualization
│       └── cli.py          # Command-line interface
├── scripts/                # Executable scripts
│   ├── detect_image.py     # Script to detect objects in an image
│   └── detect_webcam.py    # Script for real-time webcam detection
├── tests/                  # Test directory
│   ├── __init__.py
│   ├── test_model.py
│   └── test_visualization.py
└── data/                   # Data directory
    ├── images/             # Sample images
    └── outputs/            # Detection results
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/detr_vision_project.git
cd detr_vision_project
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Detecting Objects in an Image

```bash
python scripts/detect_image.py data/images/your_image.jpg --threshold 0.7 --display
```

Options:
- `--model`: DETR model to use (default: "facebook/detr-resnet-50")
- `--threshold`: Confidence threshold for detections (default: 0.7)
- `--output`: Path to save the output image
- `--device`: Device to run the model on (cpu or cuda)
- `--display`: Display the detection results

### Real-time Webcam Detection

```bash
python scripts/detect_webcam.py --threshold 0.5
```

Options:
- `--model`: DETR model to use (default: "facebook/detr-resnet-50")
- `--threshold`: Confidence threshold for detections (default: 0.5)
- `--camera-id`: Camera ID to use (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--device`: Device to run the model on (cpu or cuda)
- `--save-path`: Directory to save captured frames with detections

During webcam detection:
- Press 'q' to quit
- Press 's' to save the current frame

## Running Tests

```bash
python -m unittest discover
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook's DETR model: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
- Hugging Face Transformers: [https://huggingface.co/facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)
