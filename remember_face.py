import cv2
import face_recognition
import os
import time
import mediapipe as mp

# Mediapipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to get face encoding safely
def get_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            return encodings[0]  # Return the first encoding
        else:
            print(f"No face detected in {image_path}")
            return None  # No face found
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Directory where images are stored
image_directory = os.path.join(os.getcwd(), "images")

# List of image paths
image_paths = [
    os.path.join(image_directory, "person1.jfif"),
    os.path.join(image_directory, "person1.1.jpg"),
    os.path.join(image_directory, "person1.2.jpg"),
    os.path.join(image_directory, "person1.3.jpg"),
    os.path.join(image_directory, "person1.4.jpg"),
    os.path.join(image_directory, "person1.5.jpg"),
]

# Initialize lists for known face encodings and names
known_face_encodings = []
known_face_names = []

# Load encodings and names
for path in image_paths:
    encoding = get_face_encoding(path)
    if encoding is not None:  # Check if encoding is valid
        known_face_encodings.append(encoding)
        known_face_names.append("parn")  # Replace "parn" with the person's actual name

# Initialize the video capture
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use OpenCV backend for Windows

# Set FPS and frame scale
video_capture.set(cv2.CAP_PROP_FPS, 30)
frame_scale = 0.5  # Scale down to 50% for performance

# Initialize Mediapipe Face Detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        start_time = time.time()

        # Capture a frame
        ret, frame = video_capture.read()

        # Check if the frame is read correctly
        if not ret:
            print("Failed to capture image")
            break

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

        # Convert the frame to RGB (Mediapipe expects RGB images)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces using Mediapipe
        results = face_detection.process(rgb_frame)

        # Initialize a list to store face locations for Face Recognition
        face_locations = []
        face_names = []

        # Process Mediapipe results
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = small_frame.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                # Convert bounding box to face_recognition format (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))

        # Encode faces in the detected locations
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare each face found in the frame with the known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, assign the name of the person
            if any(matches):
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Scale bounding box back to the original frame size
            top = int(top / frame_scale)
            right = int(right / frame_scale)
            bottom = int(bottom / frame_scale)
            left = int(left / frame_scale)

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Display the frame with detected faces and names
        cv2.imshow("Video", frame)

        # Measure and print the processing time
        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.2f} seconds")

        # Break the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
