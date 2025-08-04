# DETR Vision Project

A computer vision project for object detection using Facebook's DETR (DEtection TRansformer) model.

## Features

- Object detection on images
- Real-time object detection from webcam
- Clean, modular, and well-documented code
- Command-line interface
- Support for GPU acceleration
- Modern Python packaging with `uv`

## Project Structure

```
detr_vision_project/
├── README.md               # Project documentation
├── pyproject.toml          # Project configuration and dependencies
├── .gitignore              # Files to ignore in version control
├── src/                    # Source code directory
│   └── detr_vision/        # Main package
│       ├── __init__.py     # Makes the directory a package
│       ├── model.py        # DETR model loading and inference
│       ├── camera.py       # Webcam access and processing
│       ├── visualization.py # Result visualization
│       └── cli.py          # Command-line interface
├── scripts/                # Executable scripts
│   ├── dev.sh              # Development helper script
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

## Prerequisites

- Python 3.8.1 or higher
- `uv` package manager (install with: `pip install uv`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/detr_vision_project.git
cd detr_vision_project
```

2. Install dependencies using `uv`:
```bash
# Install all dependencies
uv sync

# Or install with development dependencies
uv sync --extra dev
```

3. Activate the virtual environment (optional, `uv run` handles this automatically):
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

## Development

This project uses `uv` for dependency management and includes a convenient development script:

```bash
# Make the script executable (first time only)
chmod +x scripts/dev.sh

# Install dependencies
./scripts/dev.sh install

# Install development dependencies
./scripts/dev.sh install-dev

# Run tests
./scripts/dev.sh test

# Run tests with coverage
./scripts/dev.sh test-cov

# Format code
./scripts/dev.sh format

# Lint code
./scripts/dev.sh lint

# Run all checks (format, lint, test)
./scripts/dev.sh check

# Add a new package
./scripts/dev.sh add package_name

# Add a new development package
./scripts/dev.sh add-dev package_name

# Remove a package
./scripts/dev.sh remove package_name

# Run any command in the virtual environment
./scripts/dev.sh run python your_script.py
```

## Usage

### Detecting Objects in an Image

```bash
uv run python scripts/detect_image.py data/images/your_image.jpg --threshold 0.7 --display
```

Options:
- `--model`: DETR model to use (default: "facebook/detr-resnet-50")
- `--threshold`: Confidence threshold for detections (default: 0.7)
- `--output`: Path to save the output image
- `--device`: Device to run the model on (cpu or cuda)
- `--display`: Display the detection results

### Real-time Webcam Detection

```bash
uv run python scripts/detect_webcam.py --threshold 0.5
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

## Running Tests

```bash
python -m unittest discover
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook's DETR model: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
- Hugging Face Transformers: [https://huggingface.co/facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)