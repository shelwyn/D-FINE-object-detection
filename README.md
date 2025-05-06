# D-FINE Object Detection System

A real-time object detection implementation that leverages the D-FINE (Fine-grained Distribution Refinement) model architecture. This repository demonstrates how to use state-of-the-art transformer-based object detection for both static images and live camera feeds.

## About D-FINE

D-FINE is a powerful object detection model that employs Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD) techniques to achieve high precision in object localization. It represents an advancement over traditional DETR (DEtection TRansformer) architecture by refining probability distributions iteratively rather than predicting fixed coordinates.

## Features

- Object detection on static images with visualized bounding boxes
- Real-time object detection from webcam feed
- High accuracy detection with configurable confidence threshold
- Color-coded bounding boxes and labels for different object categories
- Support for 80+ COCO dataset object categories

## Installation

```bash
# Clone this repository
git clone https://github.com/shelwyn/D-FINE-object-detection.git
cd D-FINE-object-detection

# Create and activate virtual environment
python -m venv dfine_venv

# On Windows:
dfine_venv\Scripts\activate

# On macOS/Linux:
# source dfine_venv/bin/activate

# Clone the D-FINE repository
git clone https://github.com/Peterande/D-FINE.git

# Install D-FINE requirements
cd D-FINE
pip install -r requirements.txt
cd ..

# Install the latest transformers library from GitHub
pip install git+https://github.com/huggingface/transformers.git

# Install other required packages
pip install opencv-python pillow
