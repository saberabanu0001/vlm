# Real-Time VLM Webcam Captioning
## 📝 Description
This project demonstrates a real-time computer vision application that uses a Visual-Language Model (VLM) to generate descriptive text captions for a live webcam stream. The system employs aggressive frame-skipping and optimized data handling to run a large language model (BLIP) smoothly on a local machine, achieving near real-time scene understanding.

The **VLM** is designed to act as the "eyes" and "voice" of an intelligent agent (like a personal assistant robot), describing the environment and events as they occur.

🚀 Features

- ✅ Real-Time Acquisition:
- - Streams video via OpenCV (cv2.VideoCapture(0)).

- ✅ VLM Integration:
 - - Uses Hugging Face transformers to load and run the BLIP model for image-to-text captioning.

- ✅ Performance Optimization:
 - - Processes only 1 out of every 60 frames (~2 s at 30 FPS) to prevent lag.

- ✅ Zero-Gradient Inference:
- - Employs torch.no_grad() for faster inference and lower memory usage.

- - ✅ Live Caption Overlay:
 - - Displays the generated caption directly on the video feed window in real time.



 ## 🛠️ Prerequisites
- System Requirements

***Python 3.8 +***

- A working webcam

***GPU (optional, but strongly recommended)***

- Install Dependencies
***pip install opencv-python torch transformers Pillow numpy***