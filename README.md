# yolov3-tf-api
This is People Counter System using YOLO v3 in flask (server side) and a basic frontend in html/css/js (client side).
## Acknowledgment
This work is based upon the work done by [Zihao Zhang ](https://github.com/zzh8829/yolov3-tf2).
## What is YOLO?
You only look once (YOLO) is a state-of-the-art, real-time object detection system. 
On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev.
YOLO v3 is the third iteration of the YOLO algorithm implemented in Darknet[View the original documentation ](https://pjreddie.com/darknet/yolo).
## How does it work ?
basically the backend (in flask) contains our model and the frontend is the client side that we use to implement the algorithm.
The system sends either a screenshot using the user's camera or a picture uploaded by the user.
Then sends this picture to the backend which runs the YOLO model and returns the result after counting the number of people detected by the algorithm.
This client side then displays the number of people and the result pic.
## Installation Guide
```bash
git clone https://github.com/peternabil/yolov3-tf-api.git
cd yolov3-tf-api
pip install -r requirements.txt
python app.py
```
Then open the index.html file in the client side folder in your browser
## Detection
You Have two options either to use the webcam or upload your image
[Layout ](imgs/layout.png)
## Results
[Example ](imgs/example.jpg)
[Result ](imgs/res.jpg)
[Result](imgs/layout-res.png)
