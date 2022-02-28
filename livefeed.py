from ast import parse
from flask import Flask, render_template, Response,request
import cv2
from time import sleep
from deepface import DeepFace
from deepface import detectors
from deepface.detectors import FaceDetector
import torch
import json
cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'vp80')
# video=cv.VideoWriter('static/myvideo.webm',fourcc ,6,(320,240))
currentframe=0
model_name ='VGG-Face'
detector_backend = "opencv"
model = DeepFace.build_model(model_name)
face_detector = FaceDetector.build_model(detector_backend)
gun_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')   # or yolov5m, yolov5l, yolov5x, custom
scissor_model = torch.hub.load('ultralytics/yolov5', 'custom', path='gun_model/best.pt') # or yolov5m, yolov5l, yolov5x, custom
app = Flask(__name__, static_folder='static')
@app.route('/')
def index():
    	return render_template('index.html')
def gen():
	while(cap.isOpened()):
		ret, frame = cap.read()
		try:
			if ret == True:
				frame = cv2.resize(frame, (0,0), fx=1, fy=1)
				faces = FaceDetector.detect_faces(face_detector, detector_backend, frame, align = False)
				if (len(faces)):
					face, (x, y, w, h) = faces[0]
					recognitiondf = DeepFace.find(img_path = frame, db_path = "db", model=model, detector_backend = 'opencv', enforce_detection=False)
					person = recognitiondf.values[0][0].replace('.jpg', '').replace('db/', '') if recognitiondf.values.any() else ""
					#print(person)
					start_point = (x, y)
					end_point = (x + w, y + h )
					color = (255, 255, 255)
					thickness = 2
					image = cv2.rectangle(frame, start_point, end_point, color, thickness)
					cv2.putText(image, person, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)

				scissor_result = scissor_model(frame)
				scissor_result.pandas().xyxy[0]
				parsed_scissor_result = scissor_result.pandas().xyxy[0].to_json(orient = "records")
				parsen_json_result = json.loads(parsed_scissor_result)

				if len(parsen_json_result):
					parsed_result = parsen_json_result[0]
					start_scissor_point = (int(parsed_result['xmin']), int(parsed_result['ymin']))
					end_scissor_point = (int(parsed_result['xmax']), int(parsed_result['ymax']))
					color = (255, 255, 255)
					thickness = 2
					new_frame = cv2.rectangle(frame, start_scissor_point, end_scissor_point, color, thickness)
					cv2.putText(new_frame, parsed_result['name'], (int(parsed_result['xmin']), int(parsed_result['ymin']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

				results = gun_model(frame)
				results.pandas().xyxy[0]
				parsed = results.pandas().xyxy[0].to_json(orient = "records")
				parsed_result = json.loads(parsed)
				if len(parsed_result):
					print(parsed_result)
					parsed = parsed_result[0]
					print('print parsed:',parsed)
					start_point = (int(parsed['xmin']), int(parsed['ymin']))
					end_point = (int(parsed['xmax']), int(parsed['ymax'] ))
					color = (255, 255, 255)
					thickness = 2
					frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
					cv2.putText(frame, parsed['name'], (int(parsed['xmin']), int(parsed['ymin']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
				frame = cv2.imencode('.png', frame)[1].tobytes()
			else:
				frame = cv2.imencode('.png', frame)[1].tobytes()
		except Exception as ex:
			print(ex)
			frame = cv2.imencode('.png', cv2.imread('white-noise.jpg'))[1].tobytes()

		yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
		sleep(0)
@app.route('/',methods=['POST'])
def getval():
	k=request.form['psw1']
	if k=='2':
		cap.open(0)
		return render_template("index.html")
	if k=='1':
		cap.release()
		return render_template("index.html")
	if k=='0':
		return render_template("saved.html")
@app.route('/video_feed')
def video_feed():
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    	app.run(host='0.0.0.0', debug=False, threaded=True)
