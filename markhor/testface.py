import face_recognition #face detection
import imutils
import pickle
# import time
import cv2
# from itertools import count
# import pandas as pd
import matplotlib.pyplot as plt
import cv2 #for image processing 
import cv2 as cv  #for image processing 
from mail import SendMail #to sendmail after detation
import tkinter as tk #to create GUI (graphical user unterface)
from tkinter import *
#from tkinter import messagebox
import numpy as np
from PIL import Image,ImageTk
import time
import playsound 


# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# # Min Height and Width for the  window size to be recognized as a face
# minW = 0.1 * cap.get(3)
# minH = 0.1 * cap.get(4)
whT = 320
confThreshold =0.5 #weapon confidence
nmsThreshold= 0.0
#cap = cv2.VideoCapture('WhatsApp Video 2021-05-31 at 8.42.24 PM.mp4')
plt.rcParams['figure.figsize'] = [12, 5]


window = tk.Tk() #creating an object for tkinter
window.title("Face-Weapon-detection-based-security-System")

window.geometry('1000x600') #providing height and width
#window.configure(background='grey')
img = Image.open('background.jpg') #to add image to bakcground
bg = ImageTk.PhotoImage(img)
label1 = Label(window, image=bg,bg = 'black')
label1.place(x= 0,y=0)
classesFile = "files-req/classes.names"
classNames = []
id = 0

#reading all the classes names from the file
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files

modelConfiguration = "files-req/yolov3_custom.cfg"
modelWeights = "yolov3_custom_4600.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

count = 0
Flag = True
def findObjects(outputs,img):
    global count , Flag
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:

        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        print(classNames[classIds[i]])
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        count += 1
        print('//'*10,count,'\\'*10)
        if count%10 == 0.0:


            #Flag = False
            playsound.playsound('siren2.mp3')
            tk.messagebox.showwarning(title='Security Alert No. {}'.format(int(count/10)), message='A weapon is detected within the premises: Continue to send mail')

            SendMail(img)







def camera(img):
    blob = cv.dnn.blobFromImage(img,1/255,(whT,whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    #cv.imshow('video',img)

# initialize 'currentname' to trigger only when a new person is identified
currentname = "unknown"
# determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# use this xml file
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

detector = cv2.CascadeClassifier(cascade)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
count = 0
def getface():
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown" # if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # if someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):

            # draw the predicted face name on the image - color is in BGR
            y = top - 15 if top - 15 > 15 else top + 15
            if name == 'asfar':
                cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 225), 2)

                cv2.putText(frame, 'Asfar', (left, y), cv2.FONT_HERSHEY_SIMPLEX,.8, (225, 0, 0), 2)
                flag = True
            else:
                #cv2.putText(frame,name,(left,y), cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,225),2)
                camera(frame)


        # display the image to our screen
        cv2.imshow("OUTPUT", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit when 'q' key is pressed
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

# stop the timer and display FPS information

# start the FPS counter


def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit detection?"):
        window.destroy()
        
window.protocol("WM_DELETE_WINDOW", on_closing)

message = tk.Label(window, text="Face-Weapon-detection-based-security-System",
                   font=('incised', 18, 'bold '),bg = 'grey',fg='white')
message.place(x=200, y=70)
Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                      height=3, font=('incised', 17, 'bold'))

FA = tk.Button(window, text="Start Live Detection",fg="white",command=getface  ,bg="blue2"  ,width=20  ,height=3, activebackground = "white" ,font=('incised', 15, ' bold '))
FA.place(x=10, y=500)

FA = tk.Button(window, text="Stop Live Detection",fg="white",command=on_closing  ,bg="red"  ,width=20  ,height=3, activebackground = "white" ,font=('incised', 15, ' bold '))
FA.place(x=740, y=500)

# quitWindow = tk.Button(window, text="Manually Fill Attendance", command=manually_fill  ,fg="black"  ,bg="skyblue"  ,width=20  ,height=3, activebackground = "blue" ,font=('times', 15, ' bold '))
# quitWindow.place(x=990, y=500)

window.mainloop()
cv2.destroyAllWindows()


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()