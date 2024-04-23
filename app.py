import matplotlib
matplotlib.use("Agg")
from tensorflow.keras import models
from keras.models import load_model, model_from_json
from collections import deque
import numpy as np
import pickle, sys, imutils
import cv2
import numpy as np
import imutils
import sys
import cv2
from flask import Flask, request, render_template,Response
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    """Video streaming home page."""
    if request.method=="POST":
        path = request.form.get("path")
        return render_template('result.html',path1=path)
    else:
        return render_template('index.html')

def gen(path):
    path_to_input_video = path
    print(path)
    print("[INFO] loading model")
    with open("ucf101cnnmodel.json", 'r') as model_json:
        model = model_from_json(model_json.read())
    model.load_weights("ucf101cnnmodel.hd5")

    CLASSES = open("action_label.txt").read().strip().split("\n")

    skip = False
    color = True
    img_height,img_width,img_depth=32,32,10

    cap = cv2.VideoCapture(path_to_input_video)
    fps = cap.get(3)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    if skip:
        frames = [x * fps / img_depth for x in range(img_depth)]
    else:
        frames = [x for x in range(img_depth)]

    framearray = []

    for i in range(img_depth):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_height, img_width))

        if color:
            framearray.append(frame)
        else:
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    input = np.array(framearray)
    #print(input.shape)

    test = np.rollaxis(np.rollaxis(input,2,0),2,0)
    #print(test.shape)

    prediction = model.predict(np.expand_dims(test, axis=0))[0]
    label = CLASSES[np.argmax(prediction)]
    print("Activity :", label)

    frames = []
    vs = cv2.VideoCapture(path_to_input_video)
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=400)
        frames.append(frame)
    c=0
    while True:
        for frame in frames:
            c=c+1
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            framenext=cv2.putText(frame,label,(10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
            img = cv2.resize(framenext, (0,0), fx=1.2, fy=1.2) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            time.sleep(0.05)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

@app.route('/video_feed/<string:path>')
def video_feed(path):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

