import os
import cv2

# Define the directory to store the data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the dataset size
number_of_classes = 3
dataset_size = 100

# Initialize the camera (try different indexes if the camera is not found)
camera_index = 0  # Start with index 0
cap = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video device at index {camera_index}.")
    # Try other indexes if necessary
    for index in range(1, 4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}.")
            break
    else:
        print("Error: Could not find any available camera.")
        exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to press 'q' to start collecting data
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if not ret:
        print("Error: Camera read failed, exiting.")
        break

    # Start collecting dataset_size number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
