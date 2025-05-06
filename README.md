# D-FINE-object-detection
A real-time object detection implementation using the D-FINE (Fine-grained Distribution Refinement) model from Hugging Face's transformers library. This repository demonstrates how to use state-of-the-art transformer-based object detection models for both static images and live video feeds.


Object detection on static images with bounding box visualization
Real-time object detection from webcam feed
Support for the D-FINE model, which offers excellent accuracy and speed
Performance metrics display (FPS counter for video)
Configurable detection confidence threshold
Color-coded bounding boxes and labels for different object categories

Requirements

Python 3.9+
PyTorch
Hugging Face Transformers (latest version from GitHub)
OpenCV (for camera feed)
PIL/Pillow (for image processing)

Background
D-FINE is a powerful object detection model that employs Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD) to achieve high localization precision. It improves upon the DETR (DEtection TRansformer) architecture by transforming the regression process from predicting fixed coordinates to iteratively refining probability distributions
