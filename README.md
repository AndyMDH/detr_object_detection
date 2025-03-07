# My Object Detector

A simple object detection project using Facebook's DETR model.

## Description

This project uses a pre-trained DETR (DEtection TRansformer) model to detect objects in images and video streams. It's built as a learning project to understand computer vision and deep learning.

## Features

- Object detection on images
- Real-time object detection using webcam
- Visualization of detection results

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/my-object-detector.git
cd my-object-detector
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
```
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

4. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

### Run Object Detection on Images

```
python scripts/run_image_detection.py --image path/to/image.jpg
```

### Run Object Detection on Webcam

```
python scripts/run_camera_detection.py
```

Press 'q' to quit the webcam application.

## Project Structure

- `src/`: Contains the source code for the object detection system
- `scripts/`: Contains executable scripts to run the project
- `images/`: Directory for test images

## License

This project is for educational purposes only.

## Acknowledgments

- Facebook AI Research for the DETR model
- Hugging Face for model distribution