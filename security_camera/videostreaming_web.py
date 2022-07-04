#from calendar import day_abbr, month
import encodings
from flask import Flask, Response, render_template
#from pytz import HOUR
from sklearn.preprocessing import scale
from venv.motion import SingleMotionDetector
from imutils.video import VideoStream
import face_recognition
import cv2
import pickle
import threading
import argparse
import datetime
import torch 

import imutils
from imutils.object_detection import non_max_suppression
import time
import numpy as np 

# initialize the output frame and a lock used to ensure thread-safe exchanges
outputFrame = None
lock = threading.Lock()


app = Flask(__name__)
url = "rtsp://quebuennombre:1234cameraparacv2@192.168.45.19:554/stream1"
#"https://quebuennombre:1234cameraparacv@192.168.45.159:8080/video"

#cap = cv2.VideoCapture(url)
cap = VideoStream(url).start()
time.sleep(2.0)
# cap.set(cv2CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 15)


@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and lock variables
	global cap, outputFrame, lock

	#-------------Import models to use
	md = SingleMotionDetector(accumWeight=0.1)
	face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	model = torch.hub.load('ultralytics/yolov5', 'custom', path=args["ylmodel"], force_reload=True)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	data = pickle.loads(open(args["encodings"], "rb").read())
	#------------Some important variables
	total = 0 # loop over frames from the video stream
	count=0 # in order to save unknown detected faces

	while True: 
		# read the next frame from the video stream, resize it, convert the frame to grayscale, and blur it
		frame = cap.read()
		frame = imutils.resize(frame, width=448)
		rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		r = frame.shape[1]/float(rgb.shape[1])

		# grab the current timestamp a
		timestamp = datetime.datetime.now()

		# put it on the frame
		# cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


	#--------------------------Person detection--------------------------------------------

		#get the people. Constructs an image pyramid with scale=1.05 and a sliding window step size of (4, 4) pixels in both the x and y direction
		(rects, weights) = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)

		for (x, y, w, h) in rects: 
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
		#apply non-maxima suppresion
		rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.50)

		#draw the final suppressed square
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA, yA), (xB, yB), (255,0,0), 2)


	#--------------------------Crime detection--------------------------------------------
		
		model.to(device)
		classes = model.names
		print(classes)
		# frame = [frame]
		results = model(frame)
		print(results)
		labels, cord = results.xyxyn[0][:, -1].cpu().data.numpy(), results.xyxyn[0][:, :-1].cpu().data.numpy()
		n = len(labels)
		x_shape, y_shape = frame.shape[1], frame.shape[0]
		for i in range(n):
			row = cord[i]
			if row[4] >= 0.3:
				x1,y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
				bgr = (0, 255, 255)
				cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
				cv2.putText(frame, classes[int(labels[i])], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)


	#--------------------------Face detection and face recognition------------------------

		# faces = face_detector.detectMultiScale(rgb, 1.3, 5)
		# for(x,y,w,h) in faces:
		# 	cv2.rectangle(frame, (x, y), (x + w, y + h ), (255, 0 ,0), 2)

		boxes = face_recognition.face_locations(rgb, model= "cnn")
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		for encoding in encodings:
			matches = face_recognition.compare_faces(data["encodings"], encoding)
			name = "Unknown"

			if True in matches:
				matchedIdxs = [i for (i,b) in enumerate(matches) if b]
				counts = {}

				for i in matchedIdxs:
					name= data["names"][i]
					counts[name] = counts.get(name, 0)+1
				name = max(counts, key=counts.get)


			names.append(name)

		#Loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
		#box coordinated for detection
			top = int(top*r)
			right = int(right*r)
			bottom = int(bottom*r)
			left = int(left*r)

		#Save the information of the picture
			day = timestamp.day
			month = timestamp.month
			year = timestamp.year
			hour = timestamp.hour
			minute = timestamp.minute

			if name == 'Unknown':
				count+=1
				if count %10 == 0:
					face = frame[top:bottom, left:right]
					face_r = imutils.resize(face, width=400, height=400, inter=cv2.INTER_CUBIC)
					cv2.imwrite('C:/Users/juana/Desktop/sec_app/'+ 'Unknown_{}_{}_{}_{}_{}_{}.jpg'.format(count, day, month, year, hour, minute), face_r)
				else:
					continue

			#draw the predcited face name
			cv2.rectangle(frame, (left,top), (right, bottom), (0,255,0), 2)
			y = top - 15 if top -15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)


			#TO DO: 
			# PLAY WITH GAMMA VALUES
			# EXTEND THE BOX COORDINATES OF THE SAVED FACE? USE THE FACE DETECTION COORDINATES?

		#--------------------------Motion detection-------------------------

		#if the total number of frames has reached a sufficient number to construct a reasonable background model,
        #then continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)

			# cehck to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
		# update the background model and increment the total number of frames read thus far
		md.update(gray)
		total += 1

		# acquire the lock, set the output frame, and release the lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

# def face_detection():
# 		# grab global references to the video stream, output frame, and lock variables
# 	# global cap, outputFrame, lock

# #colocamos el codigo hecho anteriormente
# def generate():
#     while True:
# 	face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#         ret, frame = cap.read()
#         frame = cv2.resize(frame,(int(240),int(320)))
#         if ret:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_detector.detectMultiScale(gray, 1.3, 5)
#             for(x,y,w,h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h ), (255, 0 ,0), 2)
#                 (flag, encodedImage) = cv2.imencode(".jpg", frame) #comprime la imagen y la almacena en el buffer 
#                 #de memoria. Se codifica en .jpg para reducir la carga de la web y por ende, ser mas rapido
#                 if not flag: #si la imagen a sido coficada o no
#                      continue #si no, que pase a la siguiente imagenz
#                 yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)+ b'\r\n')
#                 #esto para que el browser puede consumir las imagenes



@app.route("/video_feed")
def video_feed():
    return Response(generate(),#para que el streaming se visualize en el browser
    mimetype="multipart/x-mixed-replace; boundary=frame") #mimetype dice el tipo de contenido
    #multipart/x ... es para especificar que cada imagen que venga reemplaze a la anterior

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	ap.add_argument("-m", "--ylmodel", required=True,
	help='The path to the Yolov5 model')
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
	t.daemon = True
	t.start()

	# f_t = threading.Thread(target=face_detection) #, args=(args["frame_count"],))
	# f_t.daemon = True
	# f_t.start()	

	# t.join()
	# f_t.join()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)


# if __name__ == "__main__":
#     app.run(debug=False)

# cap.release()