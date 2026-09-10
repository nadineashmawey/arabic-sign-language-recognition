import os
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "carol_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "Number of class names does not match model outputs!"
    )


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 224

# Average predictions across the last several frames
prediction_history = deque(maxlen=8)

CONFIDENCE_THRESHOLD = 0.45


# =========================================================
# PREPROCESSING
# =========================================================

def prepare_image(image):

    # OpenCV gives BGR.
    # Training images were RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    # Same method we used when evaluating Carol's model
    image = tf.image.resize_with_pad(
        image,
        IMG_SIZE,
        IMG_SIZE
    )

    # Add batch dimension:
    # (224,224,3) -> (1,224,224,3)
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    raise SystemExit


print("Camera started!")
print("Place your hand inside the box.")
print("Press Q to quit.")

WINDOW_NAME = "Arabic Sign Language Recognition"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1100,
    800
)

# =========================================================
# CAMERA LOOP
# =========================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera.")
        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape


    # =====================================================
    # CREATE REGION OF INTEREST
    # =====================================================

    box_size = int(
        min(height, width) * 0.70
    )

    center_x = width // 2
    center_y = height // 2

    x1 = max(
        0,
        center_x - box_size // 2
    )

    y1 = max(
        0,
        center_y - box_size // 2
    )

    x2 = min(
        width,
        center_x + box_size // 2
    )

    y2 = min(
        height,
        center_y + box_size // 2
    )


    hand_roi = frame[
        y1:y2,
        x1:x2
    ]


    # =====================================================
    # PREPARE IMAGE
    # =====================================================

    model_input = prepare_image(
        hand_roi
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    probabilities = model(
        model_input,
        training=False
    ).numpy()[0]


    # Save probabilities from this frame
    prediction_history.append(
        probabilities
    )


    # Average last 8 frames
    smooth_probabilities = np.mean(
        prediction_history,
        axis=0
    )


    predicted_id = int(
        np.argmax(
            smooth_probabilities
        )
    )


    confidence = float(
        smooth_probabilities[
            predicted_id
        ]
    )


    predicted_class = class_names[
        predicted_id
    ]


    # =====================================================
    # DRAW HAND BOX
    # =====================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    if confidence >= CONFIDENCE_THRESHOLD:

        result_text = (
            f"{predicted_class} - "
            f"{confidence * 100:.1f}%"
        )

    else:

        result_text = (
            f"Not confident - "
            f"{confidence * 100:.1f}%"
        )


    cv2.putText(
        frame,
        result_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "Place hand inside the box",
        (x1, max(30, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "Arabic Sign Language Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEAN UP
# =========================================================

camera.release()
cv2.destroyAllWindows()