from flask import Flask, request, jsonify
import os
import pickle
from model import cnn_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")





app = Flask(__name__)

# Global variables to hold model and other data
model = None
student_dict = None


# Route to handle initial setup
@app.route('/initial_setup', methods=['POST'])
def initial_setup():
    global model, student_dict

    # Get user input from request
    name = request.form.get('name')
    enrollment_number = request.form.get('enrollment_number')
    course = request.form.get('course')
    semester = request.form.get('semester')
    images = request.files.getlist('images')  # Assuming images are uploaded as files

    data_path = "..\\student_images\\"
    # Save images for new student
    student_images_dict = cnn_model.save_images_for_new_student(name, images, data_path)

    # Load data from MySQL
    student_dict = cnn_model.load_data_from_mysql(host, user, password, database, student_images_dict)

    # Resize images
    X, y = cnn_model.resize_images(student_images_dict, student_dict)
    X_scaled = X / 255


    # Define the path to the model.pkl file
    model_file = os.path.join('model', 'model.pkl')

    # Load the trained model
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
        # Train the model again with updated data
        loaded_model.fit(X_scaled, y, epochs=30)

        # Save model back to pickle file
        with open(model_file, 'wb') as f:
            pickle.dump(loaded_model, f)

    return jsonify({'message': 'Initial setup completed'})


# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    global model, student_dict

    # Load the trained model from the pickle file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get input image from request
    image = request.files['image']  # Assuming single image is uploaded

    # Resize the image
    resized_image = resize_image(image)

    # Use the trained model to make predictions
    prediction = model.predict(resized_image)

    # Map prediction to student name using student_dict
    student_name = student_dict[prediction]

    return jsonify({'predicted_student': student_name})


if __name__ == '__main__':
    app.run(debug=True)
