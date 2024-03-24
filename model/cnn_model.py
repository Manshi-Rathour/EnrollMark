import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import pathlib
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pymysql

from sklearn.model_selection import train_test_split
import pickle

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")



data_path = "..\\student_images\\"


# ## Saving image of new student in new folder and also appending its path in dictionary

def create_student_images_dict(data_path):
    student_images_dict = {}
    # Iterate over directories in data_path
    for student_folder in os.listdir(data_path):
        student_name = student_folder
        # Get the path to the student's images
        student_image_path = os.path.join(data_path, student_folder)
        # Check if it's a directory
        if os.path.isdir(student_image_path):
            # List image files in the directory
            image_files = list(pathlib.Path(student_image_path).glob('*'))
            # Add student name and image paths to the dictionary
            student_images_dict[student_name] = image_files
    return student_images_dict


student_images_dict = create_student_images_dict(data_path)
# student_images_dict


def save_images_for_new_student(new_student_name, image_data, data_path, student_images_dict):
    # Create a folder for the new student if it doesn't exist
    student_folder_path = os.path.join(data_path, new_student_name)
    os.makedirs(student_folder_path, exist_ok=True)
    
    # Save the images sent from the backend
    image_paths = []
    for i, image in enumerate(image_data):
        image_path = os.path.join(student_folder_path, f"image_{i+1}.jpg")
        with open(image_path, "wb") as file:
            file.write(image)
        image_paths.append(pathlib.Path(image_path))
    
    # Update student_images_dict with the new student folder path
    student_images_dict[new_student_name] = image_paths
    
    return student_images_dict


# ## Connecting to MySQL server to fetch student name with their details

def load_data_from_mysql(host, user, password, database, student_images_dict):
    try:
        # Connect to MySQL without specifying the authentication plugin
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Create a cursor object
        cursor = connection.cursor()

        # Execute a query to fetch data
        query = "SELECT * FROM students"
        cursor.execute(query)

        # Fetch data and convert it into a DataFrame
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=[col[0] for col in cursor.description])

        # Convert DataFrame to dictionary
        student_dict = df.set_index('name')['student_id'].to_dict()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        return student_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


student_dict = load_data_from_mysql(host, user, password, database, student_images_dict)


print("Student Dictionary:")
print(student_dict)


# ## Resizing images

def resize_images(student_images_dict, student_dict, X=None, y=None, image_size=(180, 180)):
    if X is None:
        X = []
    if y is None:
        y = []

    processed_files = set()
    
    # Add existing files to processed_files
    for img in X:
        processed_files.add(img)
    
    for student_name, images in student_images_dict.items():
        for image in images:
            # Check if the image file exists
            if os.path.exists(str(image)):
                # Check if the image has been processed already
                if str(image) not in processed_files:
                    # Read the image
                    img = cv2.imread(str(image))
                    
                    # Check if the image was read successfully
                    if img is not None:
                        # Resize the image
                        resized_img = cv2.resize(img, image_size)
                        
                        # Convert BGR to RGB
                        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                        
                        # Append the resized image to X
                        X.append(resized_img_rgb)
                        
                        # Append the corresponding label to y
                        y.append(student_dict[student_name])
                        
                        # Add the filename to processed_files
                        processed_files.add(str(image))
                    else:
                        print(f"Unable to read image: {image}")
                else:
                    print(f"Image already processed: {image}")
            else:
                print(f"Image file does not exist: {image}")

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


X, y = resize_images(student_images_dict, student_dict)


print("X shape:", X.shape)
print("y shape:", y.shape)


# ## Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ## Processing : scale images

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# ## Build convolutional neural network and train it


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(180, 
                                                              180,
                                                              3)),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)



num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)       




# model.evaluate(X_test_scaled,y_test)
# predictions = model.predict(X_test_scaled)
# predictions

# score = tf.nn.softmax(predictions[0])

# np.argmax(score)




# ## Save trained model to pickel file

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

