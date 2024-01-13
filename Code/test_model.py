import face_recognition
import cv2
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load the model from disk
with open("face_recognition_model.pkl", "rb") as model_file:
    model_data = pickle.load(model_file)

known_faces = model_data["known_faces"]
known_names = model_data["known_names"]

# Camera calibration parameters (adjust based on your camera setup)
focal_length = 615.0
known_face_width = 0.15

# Draw the graph for accuracy with respect to distance
def draw_accuracy_distance_graph():
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Distance'], results_df['Accuracy'], alpha=0.7, color='blue')
    plt.title('Accuracy vs Distance')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    
    # Save the graph as an image file in the same folder
    graph_filename = "accuracy_distance_graph.png"
    plt.savefig(os.path.join(os.getcwd(), graph_filename))
    
    plt.show()

# Folder containing images for testing
test_folder_path = r"C:\Users\geetr\Desktop\face_recog\Datasets"

# Create DataFrame for results or load existing one
xlsx_filename = "detection_results.xlsx"
try:
    results_df = pd.read_excel(xlsx_filename)
except FileNotFoundError:
    columns = ['Timestamp', 'Image File', 'Detected Person', 'Accuracy', 'Distance']
    results_df = pd.DataFrame(columns=columns)

def calculate_distance(face_width_in_frame):
    return (known_face_width * focal_length) / face_width_in_frame

# Increase face detection upsampling for better accuracy
def recognize_faces_in_image(image_path, frame):
    test_image = face_recognition.load_image_file(image_path)
    
    # Apply histogram equalization
    test_image = cv2.equalizeHist(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
    
    # Increase upsampling for better face detection
    face_locations = face_recognition.face_locations(test_image, number_of_times_to_upsample=2)
    
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=calculate_dynamic_tolerance(distance))
        face_distance = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            accuracy = 1 - face_distance[best_match_index]

            # Calculate distance
            face_width_in_frame = right - left
            distance = calculate_distance(face_width_in_frame)

            # Add the result to the DataFrame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results_df = results_df._append({'Timestamp': timestamp, 'Image File': os.path.basename(image_path), 'Detected Person': name, 'Accuracy': accuracy, 'Distance': distance}, ignore_index=True)

            # Add the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (left, top - 10)
            fontScale = 0.8
            fontColor = (255, 0, 0)
            thickness = 2
            lineType = 2
            cv2.putText(frame, f"{name} Detected (Accuracy: {accuracy:.2%}, Distance: {distance:.2f} meters)", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            # Rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    return frame

# Dynamic Tolerance function based on distance
def calculate_dynamic_tolerance(distance):
    if distance > 1.5:
        return 0.6
    elif 1.0 <= distance <= 1.5:
        return 0.7
    else:
        return 0.5

# Iterate through images in the test folder
for image_file in os.listdir(test_folder_path):
    image_path = os.path.join(test_folder_path, image_file)

    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        result_image = recognize_faces_in_image(image_path, np.zeros((100, 100, 3), dtype=np.uint8))  # Dummy frame for testing

        # Display the result
        cv2.imshow("Face Recognition", result_image)
        cv2.waitKey(0)

# Save DataFrame to Excel file
results_df.to_excel(xlsx_filename, index=False)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Recognize faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate distance
        face_width_in_frame = right - left
        distance = calculate_distance(face_width_in_frame)

        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=calculate_dynamic_tolerance(distance))
        face_distance = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            accuracy = 1 - face_distance[best_match_index]

            # Calculate distance again with the updated face_width_in_frame
            face_width_in_frame = right - left
            distance = calculate_distance(face_width_in_frame)

            # Add the result to the DataFrame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results_df = results_df._append({'Timestamp': timestamp, 'Image File': 'Camera 1:', 'Detected Person': name, 'Accuracy': accuracy, 'Distance': distance}, ignore_index=True)

            # Add the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (left, top - 10)
            fontScale = 0.8
            fontColor = (255, 0, 0)
            thickness = 2
            lineType = 2
            cv2.putText(frame, f"{name} Detected (Accuracy: {accuracy:.2%}, Distance: {distance:.2f} meters)", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            # Rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Draw the accuracy vs distance graph
draw_accuracy_distance_graph()

# Save DataFrame to Excel file
results_df.to_excel(xlsx_filename, index=False)

video_capture.release()
cv2.destroyAllWindows()