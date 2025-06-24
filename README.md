# Face Mask Detection

This project is a deep learning-based face mask detector built with TensorFlow, Keras, and OpenCV. It can detect whether people in images or video streams are wearing a face mask or not.

## Features
- Train a mask detector using MobileNetV2
- Real-time face mask detection using webcam
- Dataset with `with_mask` and `without_mask` images
- Pre-trained model and face detector included

## Project Structure
```
├── detect_mask_video.py         # Real-time mask detection script
├── train_mask_detector.py       # Model training script
├── mask_detector.model.h5       # Trained Keras model
├── plot.png                     # Training loss/accuracy plot
├── requirements.txt             # Python dependencies
├── dataset/                     # Dataset folder
│   ├── with_mask/               # Images with masks
│   └── without_mask/            # Images without masks
├── face_detector/               # Face detection model files
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
```

## Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
- Place your images in `dataset/with_mask` and `dataset/without_mask`.
- The dataset should be structured as shown above.

## Training
To train the mask detector model:
```bash
python train_mask_detector.py
```
- The trained model will be saved as `mask_detector.model.h5`.
- A plot of training history will be saved as `plot.png`.

## Real-Time Detection
To run real-time mask detection using your webcam:
```bash
python detect_mask_video.py
```
- Press `q` to quit the video stream.

## Pre-trained Models
- Face detector: `face_detector/deploy.prototxt` and `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
- Mask detector: `mask_detector.model.h5`

## Requirements
See `requirements.txt` for all dependencies:
- tensorflow >= 2.12.0
- keras >= 2.12.0
- imutils >= 0.5.4
- numpy >= 1.23.0
- opencv-python >= 4.7.0.72
- matplotlib >= 3.7.0
- scipy >= 1.10.0

## Acknowledgements
- Face detector model: [OpenCV DNN](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- Mask dataset: Various public sources

## License
This project is for educational and research purposes only.
