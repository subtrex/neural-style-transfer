from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from imutils import paths
import argparse
import imutils
import cv2
import time
import numpy as np
from pymongo import MongoClient
import gridfs
import io
import base64

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

client = MongoClient("mongodb+srv://subtrex:cvsZBlP8CUesQibB@cluster0.u1uwpm7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['artify'] 
images = gridfs.GridFS(db)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            images.put(file, filename=filename) # pushing uploaded image into mongodb database
            file_data = images.find_one({'filename': filename})
            image_stream = io.BytesIO(file_data.read())
            image_stream.seek(0)
            image_bytes = np.frombuffer(image_stream.read(), np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            _, buffer_1 = cv2.imencode('.jpg', image)
            input_base64 = base64.b64encode(buffer_1).decode('utf-8')
            selected_style = request.form.get('style')
            # Neural Transfer Code
            modelPath = 'static/models/'+selected_style+'.t7'
            net = cv2.dnn.readNetFromTorch(modelPath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),(103.939, 116.779, 123.680), swapRB=False, crop=False)
            net.setInput(blob)
            start = time.time()
            output = net.forward()
            end = time.time()
            output = output.reshape((3, output.shape[2], output.shape[3]))
            output[0] += 103.939
            output[1] += 116.779
            output[2] += 123.680
            output /= 255.0
            output = output.transpose(1, 2, 0)
            output = np.clip(output * 255.0, 0, 255).astype('uint8')
            _, buffer_2 = cv2.imencode('.jpg', output)
            output_base64 = base64.b64encode(buffer_2).decode('utf-8')
            
            return render_template('index.html', uploaded_file=filename, style=selected_style, output_img=output_base64, input_img=input_base64)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
