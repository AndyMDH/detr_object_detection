from setuptools import setup, find_packages

setup(
    name="detr_vision",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "transformers>=4.18.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
    ],
    python_requires=">=3.6",
    author="Your Name",
    author_email="your.email@example.com",
    description="A computer vision project for object detection using DETR",
)