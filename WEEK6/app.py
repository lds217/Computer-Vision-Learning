import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import time
import tkinter as tk
from tkinter import messagebox

# Model paths
model_paths = {
    "1": "Data/output/checkpoint/finetune/resnet_freeze1_375_0.9480000138282776.keras",
    "2": "Data/output/checkpoint/resnet50/resnet50_978_0.9919999837875366.keras",
    "3": "Data/output/checkpoint/transfer/resnet50_855_0.6840000152587891.keras"
}

# Function to display a loading screen with a progress bar
def show_loading_screen(progress):
    loading_screen = np.zeros((480, 640, 3), dtype=np.uint8)  # Black screen
    cv2.putText(loading_screen, "Loading Model, Please Wait...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw a loading bar
    bar_x_start, bar_y_start = 50, 300
    bar_width, bar_height = 540, 30
    cv2.rectangle(loading_screen, (bar_x_start, bar_y_start), (bar_x_start + bar_width, bar_y_start + bar_height), (255, 255, 255), 2)
    cv2.rectangle(loading_screen, (bar_x_start, bar_y_start), (bar_x_start + int(bar_width * progress), bar_y_start + bar_height), (0, 255, 0), -1)
    
    cv2.imshow("Webcam Prediction", loading_screen)
    cv2.waitKey(1)

# Tkinter window for model selection
def open_model_selection_ui():
    def on_model_select(model_choice):
        global selected_model_path
        selected_model_path = model_paths.get(model_choice, model_paths["1"])  # Default to Model 1
        root.destroy()  # Close the window after selection

    root = tk.Tk()
    root.title("Model Selection")

    label = tk.Label(root, text="Select a model to load:", font=("Arial", 14))
    label.pack(pady=10)

    button1 = tk.Button(root, text="1: Finetune (resnet_freeze1_0.94)", font=("Arial", 12), 
                        command=lambda: on_model_select("1"))
    button1.pack(fill='x', padx=20, pady=5)

    button2 = tk.Button(root, text="2: ResNet50 (1000 epochs, high accuracy, less data)", font=("Arial", 12), 
                        command=lambda: on_model_select("2"))
    button2.pack(fill='x', padx=20, pady=5)

    button3 = tk.Button(root, text="3: Transfer Learning (moderate accuracy)", font=("Arial", 12), 
                        command=lambda: on_model_select("3"))
    button3.pack(fill='x', padx=20, pady=5)

    root.mainloop()

# Open the model selection UI
open_model_selection_ui()

# Loading the model
print("Loading model, please wait...")
for i in range(10):  # Simulate loading steps
    show_loading_screen(i / 10)  # Update progress bar
    time.sleep(0.1)  # Simulate delay for loading parts of the model

model = load_model(selected_model_path)
show_loading_screen(1)  # Show fully loaded bar
print("Model loaded successfully.")
time.sleep(0.5)  # Pause to show the complete loading bar

# Load class indices from label.txt
class_dict = {}
try:
    with open("label.txt", "r") as f:
        for line in f:
            index, class_name = line.strip().split(": ")
            class_dict[int(index)] = class_name
    print("Class dictionary loaded successfully.")
except FileNotFoundError:
    print("Error: label.txt file not found.")
    exit()

# Function to load the first image from a class folder
def load_reference_image(class_name):
    class_folder_path = os.path.join("Data/output/test", class_name)
    
    if os.path.isdir(class_folder_path):
        image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            ref_image_path = os.path.join(class_folder_path, image_files[0])
            ref_image = cv2.imread(ref_image_path)
            if ref_image is not None:
                ref_image = cv2.resize(ref_image, (100, 100))  # Resize to fit in the top right corner
                return ref_image
    
    return None

# Open a connection to the webcam, set a smaller frame size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press any key to exit the program.")
instruction_text = "Press any key to exit"

reference_image = None
predicted_class_name = None
last_update_time = time.time()  # Track the last time the reference image was updated

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the frame for prediction
        frame_resized = cv2.resize(frame, (224, 224))  # Resize for model input
        input_arr = img_to_array(frame_resized)  # Convert to array
        input_arr = input_arr * 1./255  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Make a prediction
        result = model.predict(input_arr)
        idx = np.argmax(result)
        new_predicted_class_name = class_dict.get(idx, "Unknown")

        # Update the reference image if a new class is detected and 0.5 seconds have passed
        current_time = time.time()
        if new_predicted_class_name != predicted_class_name and (current_time - last_update_time) > 0.5:
            predicted_class_name = new_predicted_class_name
            reference_image = load_reference_image(predicted_class_name)
            last_update_time = current_time  # Reset the timer

        # Add a semi-transparent overlay for text display
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Display the prediction and instructions on the frame
        cv2.putText(frame, f"Predicted: {predicted_class_name}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Overlay the reference image in the top right corner, if available
        if reference_image is not None:
            frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = reference_image

        # Show the frame
        cv2.imshow("Webcam Prediction", frame)

        # Exit the loop if any key is pressed
        if cv2.waitKey(1) != -1:
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted")

cap.release()
cv2.destroyAllWindows()
