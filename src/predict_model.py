import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from unet_model import dice_loss
from load_data import normalize_image, image_to_array
from utils import load_config

CONFIG_PATH = "./config.yaml"


def preprocess_test_data(images):
    processed_images = normalize_image(images)
    processed_images = tf.image.resize(processed_images, (128, 128)).numpy()
    return processed_images


def show_images_and_masks(images, masks):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title('Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    config = load_config(CONFIG_PATH)

    # Load the model
    model = models.load_model(config["save_model_path"], custom_objects={'dice_loss': dice_loss})

    # Load and preprocess test data
    test_data = pd.read_csv(config["test_dataset_path"]).sample(5)
    images_id = list(test_data['ImageId'])
    # images_id = ["0b7359c38.jpg"] # Goog image
    images = np.array([image_to_array(config["test_images_path"] + "/" + image_id) for image_id in images_id])

    # Preprocess images
    processed_images = preprocess_test_data(images)

    # Model prediction
    predicted_masks = model.predict(processed_images)

    # Model predictions visualization
    show_images_and_masks(images, predicted_masks)


main()

