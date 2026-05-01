import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

model = tf.keras.models.load_model("cancer_model.h5")

# IMPORTANT: match class order from training output
class_names = ['Melanoma', 'NotMelanoma']

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    class_index = np.argmax(preds)
    label = class_names[class_index]
    confidence = preds[class_index]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

    return label, confidence


if __name__ == "__main__":
    image_path = "DermMel/test/Melanoma/AUG_0_11.jpeg"
    predict_image(image_path)