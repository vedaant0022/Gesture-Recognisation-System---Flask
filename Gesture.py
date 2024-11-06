

from flask import Flask, jsonify, Response, render_template, request
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
from flask_cors import CORS
import random
import smtplib
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bson import ObjectId
import cloudinary
import cloudinary.uploader
import os

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize mediapipe and gesture recognizer model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().splitlines()

# Initialize webcam and current gesture
cap = cv2.VideoCapture(0)
current_prediction = None

# MongoDB initialization
client = MongoClient(os.getenv('MONGO_URI'))
db = client['otp_auth']
otp_collection = db['otps']
verified_users_collection = db['verified_users']  # New collection for verified users

# Email configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))

# Image upload configuration
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cloudinary configuration
# Configure Cloudinary with your account credentials
cloudinary.config(
    cloud_name= os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key= os.getenv("CLOUDINARY_API_KEY"),
    api_secret= os.getenv("CLOUDINARY_API_SECRET")
)

# Helper function to send email
def send_email(receiver_email, otp):
    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            message = f'Subject: Your OTP Code\n\nYour OTP is: {otp}. It is valid for 5 minutes.'
            server.sendmail(EMAIL_USER, receiver_email, message)
    except Exception as e:
        print(f"Error sending email: {e}")

# Generate OTP and send it to the user's email
@app.route('/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = str(random.randint(100000, 999999))
    otp_data = {
        "email": email,
        "otp": otp,
        "expires_at": datetime.utcnow() + timedelta(minutes=5)
    }
    otp_collection.update_one({"email": email}, {"$set": otp_data}, upsert=True)
    send_email(email, otp)

    return jsonify({"message": "OTP sent to email"}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server started"}), 200

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')

    if not email or not otp:
        return jsonify({"error": "Email and OTP are required"}), 400

    # Retrieve OTP from MongoDB
    otp_record = otp_collection.find_one({"email": email})

    if otp_record:
        # Check if OTP is correct and not expired
        if otp_record["otp"] == otp and otp_record["expires_at"] > datetime.utcnow():
            otp_collection.delete_one({"email": email})  # OTP verified, remove it from the database
            
            # Store the verified email in the verified_users_collection with an empty gestures list
            verified_user_data = {
                "email": email,
                "verified_at": datetime.utcnow(),  # Store the time of verification
                "gestures": []  # Initialize an empty list for gestures
            }
            verified_users_collection.update_one(
                {"email": email},
                {"$set": verified_user_data},
                upsert=True  # Insert if the email does not exist
            )

            return jsonify({"message": "OTP verified successfully, email stored."}), 200
        else:
            return jsonify({"error": "Invalid or expired OTP"}), 400
    else:
        return jsonify({"error": "OTP not found"}), 404
    
@app.route('/getUserById', methods=['POST'])
def get_user_by_id():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        # Convert the user_id to an ObjectId and fetch the user document
        user_record = verified_users_collection.find_one({"_id": ObjectId(user_id)})

        if user_record:
            # Convert ObjectId to string for JSON compatibility
            user_record['_id'] = str(user_record['_id'])
            return jsonify(user_record), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": f"Invalid user ID format: {str(e)}"}), 400

    
    
@app.route('/getDetails', methods=['POST'])
def get_details():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Find the user in the verified_users collection
    user_record = verified_users_collection.find_one({"email": email}, {"_id": 1})

    if user_record:
        return jsonify({"_id": str(user_record["_id"])}), 200
    else:
        return jsonify({"error": "User not found"}), 404
    
@app.route('/addGesture', methods=['POST'])
def add_manual_gesture():
    data = request.form
    user_id = data.get('user_id')
    gesture = data.get('gesture')
    image = request.files.get('image')

    # Check required fields
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    if not gesture:
        return jsonify({"error": "Gesture is required"}), 400
    if not image:
        return jsonify({"error": "Image is required"}), 400

    try:
        # Convert the user_id to an ObjectId
        user_object_id = ObjectId(user_id)

        # Upload the image to Cloudinary
        upload_result = cloudinary.uploader.upload(image, public_id=f"gestures/{gesture}_{image.filename}")

        # Get the URL of the uploaded image
        image_url = upload_result.get("url")

        # Create a dictionary with gesture name and Cloudinary image URL
        gesture_entry = {
            "gesture": gesture,
            "image": image_url
        }

        # Append the gesture entry to the user's gestures list in the database
        result = verified_users_collection.update_one(
            {"_id": user_object_id},
            {"$push": {"gestures": gesture_entry}}
        )

        if result.matched_count == 0:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "message": "Gesture and image URL added to user's record",
            "gesture_entry": gesture_entry
        }), 200

    except Exception as e:
        return jsonify({"error": f"Invalid user ID format or database error: {str(e)}"}), 400
    
@app.route('/getUserGestures', methods=['POST'])
def get_user_gestures():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        # Convert the user_id to an ObjectId
        user_object_id = ObjectId(user_id)

        # Find the user document by the provided ObjectId
        user_record = verified_users_collection.find_one({"_id": user_object_id}, {"gestures": 1})

        if user_record:
            # Return the gestures list
            return jsonify({"gestures": user_record.get("gestures", [])}), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": f"Invalid user ID format: {str(e)}"}), 400

def generate_frames():
    global current_prediction
    while True:
        # Read frame from the webcam
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(framergb)
            className = ''

            # Detect hand and predict gesture
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * frame.shape[1])
                        lmy = int(lm.y * frame.shape[0])
                        landmarks.append([lmx, lmy])

                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]

            current_prediction = className

            # Display the gesture name on the frame
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0,0,255), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format for the MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to get the current gesture prediction
@app.route('/gesture', methods=['GET'])
def get_gesture():
    return jsonify({"gesture": current_prediction})

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
