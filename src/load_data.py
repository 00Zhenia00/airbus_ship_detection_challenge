import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from utils import load_config

CONFIG_PATH = "config.yaml"


def normalize_image(input_image):
    return input_image / 255.0


def image_to_array(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def rle_to_image(rle: str, image_shape):
    image_mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.int16)

    rle_split = rle.split()

    starts = np.array(rle_split[::2], dtype=int)
    lengths = np.array(rle_split[1::2], dtype=int)
    ends = starts + lengths

    for i in range(len(starts)):
        image_mask[starts[i]:ends[i]] = 1

    return image_mask.reshape(image_shape).T


class AirbusDataset(Sequence):
    """
    Dataset class for Airbus ship detection data.

    """
    def __init__(self, data, images_dir, input_shape, output_shape, batch_size=16, shuffle=True, class_weights=None,
                 image_preprocess_fn=None, mask_preprocess_fn=None):
        self._df = data
        self._image_dir = images_dir
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._batch_size = batch_size
        self._batch_indices = None
        self._shuffle = shuffle
        self._class_weights = class_weights
        self._image_preprocess_fn = image_preprocess_fn
        self._mask_preprocess_fn = mask_preprocess_fn

        self.on_epoch_end()

    def __len__(self):
        """ Return amount of batches. """
        return len(self._df) // self._batch_size

    def __getitem__(self, index):
        """ Generate single batch"""

        # Get batch dataframe
        batch_indices = self._batch_indices[index]
        batch_df = self._df.iloc[batch_indices]

        # Get batch images, masks
        batch_df['Image'] = batch_df['ImageId'].map(lambda x: self._image_to_array(x))
        batch_df['Mask'] = batch_df['Masks'].map(lambda x: self._masks_as_image(x))

        images = np.array(list(batch_df['Image']), dtype=np.float32)
        masks = np.array(list(batch_df['Mask']), dtype=np.float32)

        images = tf.image.resize(images, self._output_shape).numpy()
        masks = tf.image.resize(masks, self._output_shape).numpy()

        # Preprocess images
        if self._image_preprocess_fn:
            images = self._image_preprocess_fn(images)

        # Preprocess masks
        if self._mask_preprocess_fn:
            masks = self._mask_preprocess_fn(masks)

        # DEPRECATED
        if self._class_weights:
            class_weights = self._class_weights / tf.reduce_sum(self._class_weights)
            batch_df['Weights'] = batch_df['Mask'].map(lambda x: tf.gather(class_weights, indices=tf.cast(x, tf.int32)))
            weights = np.array(list(batch_df['Weights']), dtype=np.float32)
            return images, masks, weights

        return images, masks

    def on_epoch_end(self):
        indices = np.arange(len(self._df))

        if self._shuffle:
            np.random.shuffle(indices)

        # Split indices on batches
        self._batch_indices = np.array_split(indices, np.arange(self._batch_size, len(indices), self._batch_size))

    def _masks_as_image(self, masks: list):
        all_masks = np.zeros(self._input_shape, dtype=np.int16)
        for mask in masks:
            if isinstance(mask, str):
                all_masks += self._rle_to_image(mask)
        return np.expand_dims(all_masks, -1)

    def _rle_to_image(self, rle):
        if rle != 'nan':
            return rle_to_image(rle, self._input_shape)
        return np.zeros(self._input_shape, dtype=np.int16)

    def _image_to_array(self, image_id):
        image_path = self._image_dir + '/' + image_id
        return image_to_array(image_path)

    @property
    def class_weights(self):
        return self._class_weights


def load_data(train_dataset_path, train_images_path, input_shape=(768, 768), output_shape=(256, 256), data_size=20000,
              batch_size=16, train_size=0.7):
    config = load_config(CONFIG_PATH)

    # Load train dataset
    train_data = pd.read_csv(train_dataset_path)

    # Group images by ImageId and append all masks to list
    train_data['EncodedPixels'] = train_data['EncodedPixels'].astype(str)
    train_data = train_data.groupby('ImageId')['EncodedPixels'].apply(list).reset_index(name='Masks')

    train_data['ShipsPresence'] = train_data['Masks'].apply(lambda x: 1 if x != ['nan'] else 0)

    # Split dataset on ship and non-ship datasets
    train_data_ships = train_data[train_data['ShipsPresence'] == 1]
    train_data_no_ships = train_data[train_data['ShipsPresence'] == 0]

    # Make balanced dataset
    ship_ratio = config["ship_images_ratio"]
    non_ship_ratio = config["non_ship_images_ratio"]
    train_data_balanced = pd.concat([train_data_ships.sample(int(data_size*ship_ratio)),
                                     train_data_no_ships.sample(int(data_size*non_ship_ratio))])

    # Count images with ships and without ships (DEPRECATED)
    # train_data_ships = train_data.copy()
    # train_data_ships['ShipsPresence'] = train_data_ships['Masks'].apply(lambda x: 1 if x != ['nan'] else 0)
    # without_ships_count = train_data_ships['ShipsPresence'].value_counts()[0]
    # with_ships_count = train_data_ships['ShipsPresence'].value_counts()[1]

    # Find class weights for loss function balancing (DEPRECATED)
    # class_weights = tf.constant([without_ships_count, with_ships_count])
    # class_weights = class_weights / tf.reduce_sum(class_weights)
    # class_weights = class_weights.numpy()

    # Split train dataset on train and validation datasets
    train_df, val_df = train_test_split(train_data_balanced, train_size=train_size, shuffle=True)

    # Create train and validation Airbus Datasets
    train_dataset = AirbusDataset(train_df, train_images_path, input_shape=input_shape, output_shape=output_shape,
                                  batch_size=batch_size, image_preprocess_fn=normalize_image)
    val_dataset = AirbusDataset(val_df, train_images_path, input_shape=input_shape, output_shape=output_shape,
                                batch_size=batch_size, image_preprocess_fn=normalize_image)

    return train_dataset, val_dataset

