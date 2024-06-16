# Hand Gesture Recognition and Language Translation with MediaPipe and RandomForest

This project implements a hand gesture recognition system using MediaPipe for hand landmark detection and a RandomForest classifier for gesture classification. The project includes data collection, model training, and an API for classification.

## Table of Contents
- [Installation](#installation)
- [Dataset Creation](#dataset-creation)
- [Model Training](#model-training)
- [API for Classification](#api-for-classification)
- [Results Visualization](#results-visualization)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MalikZohaibMustafa/Sign-Language-Translation.git
    cd hand-gesture-recognition
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Creation

To create a dataset of hand gestures:

1. Run the `dataset_creation.py` script:
    ```bash
    python dataset_creation.py
    ```

2. Follow the on-screen instructions to capture images for each gesture class.

## Model Training

To train the model:

1. Run the `model_training.py` script:
    ```bash
    python model_training.py
    ```

2. The script will preprocess the images, extract hand landmarks using MediaPipe, train a RandomForest classifier, and save the trained model as `model.p`.

## API for Classification

The project includes a Flask API for real-time hand gesture classification:

1. Run the `app.py` script to start the server:
    ```bash
    python app.py
    ```

2. The server will be available at `http://127.0.0.1:5000`.

### API Endpoints

- `POST /classify`: Classifies a hand gesture in the uploaded image.

    **Request:**
    - `file`: The image file containing a hand gesture.

    **Response:**
    - `predicted_character`: The predicted gesture class.

- `POST /classify_multiple`: Classifies hand gestures in multiple uploaded images and constructs a sentence.

    **Request:**
    - `files`: An array of image files, each containing a hand gesture.

    **Response:**
    - `sentence`: The constructed sentence from the predicted gesture classes. Spaces are added for "space" predictions, and the last character is deleted for "del" predictions.

## Results Visualization

The training script generates and displays a confusion matrix to visualize the model's performance.

1. The confusion matrix and classification report are printed during model training.
2. The confusion matrix is also displayed as a heatmap using Matplotlib and Seaborn.

## Usage

1. **Dataset Creation**:
    ```bash
    python dataset_creation.py
    ```

2. **Model Training**:
    ```bash
    python model_training.py
    ```

3. **Start API**:
    ```bash
    python app.py
    ```

4. **Classify an Image**:
    - Send a POST request to `http://127.0.0.1:5000/classify` with an image file containing a hand gesture.

5. **Classify Multiple Images**:
    - Send a POST request to `http://127.0.0.1:5000/classify_multiple` with an array of image files, each containing a hand gesture.

## File Structure
.
├── app.py # Flask API for classification
├── dataset_creation.py # Script for creating dataset
├── model_training.py # Script for training the model
├── data # Directory to store collected images
├── model.p # Trained model file
├── data.pickle # Pickle file containing processed data
├── requirements.txt # Required packages
└── README.md # This README file


## Requirements

- Python 3.10.9
- Flask
- scikit-learn
- Matplotlib
- Seaborn
- opencv-python==4.7.0.68
- mediapipe==0.9.0.1
- scikit-learn==1.2.0


## Contributing
Feel free to submit issues, fork the repository, and send pull requests. Contributions are welcome!

