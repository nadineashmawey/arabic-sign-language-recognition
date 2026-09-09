import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("arabic_sign_model.keras")

# Same class order used during training
class_names = [
    "ع", "ال", "ا", "ب", "ض", "د", "ف",
    "غ", "ح", "هـ", "ج", "ك", "خ",
    "لا", "ل", "م", "ن", "ق", "ر", "ص",
    "س", "ش", "ط", "ت", "ة",
    "ذ", "ث", "و", "ي", "ظ", "ز"
]

# Initialize MediaPipe Hands for hand detection and tracking
mp_hands = mp.solutions.hands

# Initialize MediaPipe drawing utilities to draw hand landmarks on the frame
mp_draw = mp.solutions.drawing_utils

# Create a Hands detector for live camera input
hands = mp_hands.Hands(
    static_image_mode=False,      # Track hands across video frames
    max_num_hands=1,              # Detect only one hand
    min_detection_confidence=0.5, # Minimum confidence for initial hand detection
    min_tracking_confidence=0.5   # Minimum confidence for tracking the hand
)

# Open the default webcam
cap = cv2.VideoCapture(0)

# Start the live camera loop
while True:
    # Read one frame from the webcam
    ret, frame = cap.read()

    # Stop the loop if the camera frame could not be read
    if not ret:
        break

# Convert the frame from BGR to RGB because MediaPipe expects RGB images
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the frame to detect and track the hand
results = hands.process(rgb)

# Check if at least one hand was detected
if results.multi_hand_landmarks:
    # Get the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

# Get the height and width of the current camera frame
h, w, _ = frame.shape

# Get the x and y coordinates of all hand landmarks
x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

# Find the bounding box around the detected hand
x_min = max(min(x_coords) - 30, 0)
x_max = min(max(x_coords) + 30, w)
y_min = max(min(y_coords) - 30, 0)
y_max = min(max(y_coords) + 30, h)

# Crop the hand from the original camera frame
hand = frame[y_min:y_max, x_min:x_max]

# Check that the hand crop is not empty
if hand.size != 0:

    # Convert the hand image from BGR to RGB
    hand_rgb = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)

    # Resize the hand image to 224x224 while preserving its aspect ratio
    hand_tensor = tf.image.resize_with_pad(
        hand_rgb,
        224,
        224
    )

    # Convert the image to float32
    hand_tensor = tf.cast(
        hand_tensor,
        tf.float32
    )

    # Add a batch dimension because the model expects a batch of images
    hand_tensor = tf.expand_dims(
        hand_tensor,
        axis=0
    )


# Make a prediction using the trained model
    predictions = model.predict(hand_tensor, verbose=0)

    # Get the index of the class with the highest probability
    predicted_index = np.argmax(predictions[0])

    # Get the predicted Arabic sign language letter
    predicted_class = class_names[predicted_index]

    # Get the confidence of the prediction
    confidence = predictions[0][predicted_index]

# Draw a bounding box around the detected hand
    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    # Create the text showing the predicted letter and confidence
    text = f"{predicted_class}: {confidence * 100:.1f}%"

    # Display the prediction on the camera frame
    cv2.putText(
        frame,
        text,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )