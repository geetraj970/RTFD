import os
import face_recognition
import pickle
import matplotlib.pyplot as plt
import time
import numpy as np

def encode_faces_in_folder(folder_path, total_photos):
    known_faces = []
    known_names = []
    photos_trained = 0

    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                face_image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(face_image, model="large")

                if len(face_encoding) > 0:
                    known_faces.append(face_encoding[0])  # Take the mean as an example
                    known_names.append(person_folder)
                    photos_trained += 1

                    # Print the progress for each photo
                    print(f"Trained {photos_trained}/{total_photos} photos")

        # Update the graph after each folder
        plot_training_progress(known_faces, known_names, f"training_graph_{person_folder}.png")
    return known_faces, known_names
    

def train_and_save_model(dataset_path, model_filename):
    total_photos = sum([len(files) for _, _, files in os.walk(dataset_path)])
    known_faces, known_names = encode_faces_in_folder(dataset_path, total_photos)

    model_data = {"known_faces": known_faces, "known_names": known_names}
    with open(model_filename, "wb") as model_file:
        pickle.dump(model_data, model_file)

def plot_training_progress(known_faces, known_names, save_path=None):
    plt.clf()  # Clear the previous plot
    
    # Set the figure size to increase the height of the PNG text
    plt.figure(figsize=(10, 6))

    x_labels = [f"{i+1}" for i in range(len(known_faces))]
    plt.plot(x_labels, known_faces, marker='o', linestyle='-', color='blue')
    plt.xlabel('Photo')
    plt.ylabel('Face Encoding Value')
    plt.title('Face Recognition Model Training Progress')

    # Add explanation to the graph
    explanation = f"This line plot shows the training progress of the face recognition model.\n\n" \
                  f"The y-axis represents the mean face encoding value, and each point on the line corresponds to a photo.\n" \
                  f"Photos are labeled with the person's name: {', '.join(known_names)}"
    plt.text(0.5, -0.2, explanation, ha='center', va='center', transform=plt.gca().transAxes, color='green')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure if save_path is provided
    else:
        plt.show()

dataset_path = r"C:\Users\geetr\Desktop\face_recog\Datasets"
model_filename = "face_recognition_model.pkl"
epochs = 1

# Create the directory to save training graphs
save_dir = r"C:\Users\geetr\Desktop\face_recog"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, epochs + 1):
    train_and_save_model(dataset_path, model_filename)

# Sleep for 1 second after training completion
time.sleep(1)
