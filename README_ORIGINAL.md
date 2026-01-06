# Real-time Person Tracking and Weapon Detection

This project implements a robust real-time system for *person tracking* and *weapon detection, designed primarily for **academic research*. It leverages a flexible pipeline that allows for the integration of various state-of-the-art object detection and tracking models.

-----

## Features

  * *Person Detection & Tracking*:
      * Utilizes advanced tracking algorithms like *ByteTrack* and *BoT-SORT*.
      * Supports multiple person detection backbones including *SSD-MobileNetV2 Lite* and *YOLOv8n*, configurable via config.yaml.
  * *Weapon Detection*:
      * Employs powerful object detection models such as *YOLOv8n with EfficientViT backbone* and *RT-DETR-L* for accurate weapon identification.
  * *Flexible Input*: Processes video streams from a specified file path or a live webcam feed, controlled through config.yaml.
  * *Real-time Visualization*: Displays a live window with processed video frames, showing detected bounding boxes, object labels (person/weapon), and unique tracking IDs for persons.
  * *Modular Design*: Easily switch between different detection and tracking models to compare performance and explore various configurations.

-----

## Technical Stack

The project is built upon a foundation of cutting-edge deep learning and computer vision libraries:

  * *PyTorch & CUDA*: For efficient model inference and GPU acceleration.
  * *Ultralytics*: For leveraging YOLOv8 models.
  * *ByteTrack*: A high-performance multi-object tracking framework.
  * *BoT-SORT*: Another robust multi-object tracking algorithm.
  * *TorchVision*: Common vision datasets, models, and image transformations.
  * *Cython\_bbox*: Optimized bounding box operations.

-----

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

  * A system with *NVIDIA GPU* and *CUDA* installed for optimal performance.
  * *Anaconda* or *Miniconda* installed for environment management.
  * *Python 3.x*

### Installation

1.  *Unzip the project folder*.
2.  *Navigate into the project directory* in your terminal:
    bash
    cd your_project_folder_name
    
3.  *Create a Conda environment* (recommended):
    bash
    conda create -n your_env_name python=3.x  # Replace 3.x with your desired Python version, e.g., 3.10
    conda activate your_env_name
    
4.  *Install PyTorch with CUDA support*. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the appropriate command for your CUDA version. For example:
    bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    (Replace cu118 with your CUDA version, e.g., cu121)

    Project is developed using:
    * *torch==2.5.1+cu121*
    * *torchaudio==2.5.1+cu121*
    * *torchvision==0.20.1+cu121*

5.  *Install other required packages*:
    bash
    pip install -r requirements.txt
    

### Configuration

Edit the config.yaml file located in the root of your project directory to customize the pipeline:

  * *Select Detection Models*: Specify which models to use for person and weapon detection.
  * *Set Model Parameters*: Adjust parameters for the chosen models.
  * *Input Path*: Provide the path to your video file or set it to 0 for live webcam feed.

### Running the Pipeline

After configuring config.yaml, execute the main script:

bash
python main.py


This will start the detection and tracking pipeline, displaying the output in a new window.

-----

## Demo Videos

You can find example processed videos showcasing the system's capabilities in the processed_videos folder.

-----

## Project Structure

```
.
├───Results/                   # Contains the Training script (RTDETR/YOLOV8_EfficientViT) along with validations 
├───ByteTrack/                 # Submodule for ByteTrack tracking
├───cython_bbox/               # Cython optimized bounding box operations
├───models/                    # Custom or pre-trained models
│   ├───person_detectors/      # Person detection model definitions (e.g., SSD-MobileNetV2 Lite, YOLOv8n implementations)
│   └───weapon_detectors/      # Weapon detection model definitions (e.g., EfficientViT-YOLOv8, RT-DETR implementations)
├───processed_videos/          # Output directory for processed demo videos
├───stages/                    # (Potentially pipeline stages or helper scripts)
├───testing_data/              # Unseen data for model evaluation and testing
├───tracker/                   
├───ultralytics/               # Integrated Ultralytics (YOLOv8) library
├───weights/                   # Pre-trained model weights
│   ├───person/                # Weights for person detection models
│   │   ├───ssd_mobilenetV2_lite/ # Weights for SSD-MobileNetV2 Lite person detector
│   │   └───yolov8n/              # Weights for YOLOv8n person detector
│   └───weapon/                # Weights for weapon detection models
│       ├───efficientvit_yolov8/  # Weights for YOLOv8 with EfficientViT backbone
│       └───rt_detr/              # Weights for RT-DETR
├───test_flop.py               # Sript to get the NO. of params & GFLOPS of the models. (run: python test_flop.py)
├───main.py                    # Main script to run the pipeline
├───config.yaml                # Configuration file for model selection, input, etc.
└───requirements.txt           # Python dependencies
```