import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
hamza_image = face_recognition.load_image_file("IMG_20220907_170135_996.jpg")
#print(face_recognition.face_encodings(hamza_image))
hamza_face_encoding = face_recognition.face_encodings(hamza_image)[0]

# Load a second sample picture and learn how to recognize it.
# arif_image = face_recognition.load_image_file("arif.jpg")
# arif_face_encoding = face_recognition.face_encodings(arif_image)[0]

# khubaib_image = face_recognition.load_image_file("khubaib.jpg")
# khubaib_face_encoding = face_recognition.face_encodings(khubaib_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    hamza_face_encoding
]
known_face_names = [
    "HAMZA KHAN"
]


count = 0
C = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
       
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"


        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        if name.lower() == 'unknown':
            count += 1
            if count >= 3:
                C += 1
                cv2.imwrite(f'unknown{C}.jpg',frame)
                count=0
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()