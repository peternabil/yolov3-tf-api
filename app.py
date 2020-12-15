from flask import Flask, render_template, Response,request,jsonify
from PIL import Image,ImageDraw,ImageFont
from io import BytesIO
import base64
from flask_cors import cross_origin
import cv2
import image
import numpy as np
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
import json

model_size = (416, 416,3)
num_classes = 80

class_name = './data/coco.names'
max_output_size = 40
max_output_size_per_class= 20

iou_threshold = 0.5

confidence_threshold = 0.5

cfgfile = 'cfg/yolov3.cfg'

weightfile = 'weights/yolov3_weights.tf'

img_path = "data/images/person.jpg"

model = YOLOv3Net(cfgfile,model_size,num_classes)
model.load_weights(weightfile)

# class_names = load_class_names(class_name)


app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>The Server Works</h1>"

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_base64_file():
    """
        Upload image with base64 format and get car make model and year
        response
    """

    data = request.get_json()

    if data is None:
      print("No valid request body, json missing!")
      return jsonify({'error': 'No valid request body, json missing!'})
    else:

      img_data = data['img']
      img_data = img_data.replace("data:image/webp;base64,","")

      im = Image.open(BytesIO(base64.b64decode(img_data)))
      im,person_num,boxes,scores, classes, nums ,class_names= image.main(im,model)
      im = Image.fromarray(im)
      buffered = BytesIO()
      im.save(buffered, format="webp")
      im_bytes = buffered.getvalue()  # im_bytes: image in binary format.
      im_b64 = base64.b64encode(im_bytes)
      im_b64 = "data:image/webp;base64," + (str(im_b64).replace("b'","")[:-1])

      print(nums)
      boxes = np.array(boxes).tolist()
      scores = np.array(scores).tolist()
      classes = np.array(classes).tolist()
      nums = np.array(nums).tolist()
      class_names = np.array(class_names).tolist()
      return jsonify(img=im_b64,person_num=person_num,boxes=boxes,scores=scores, classes=classes, nums=nums, class_names=class_names)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
