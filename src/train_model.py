import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from load_data import load_data
from unet_model import init_model
from utils import visualize_history, save_train_history, load_config

# folder to load config file
CONFIG_PATH = "config.yaml"


class DisplayCallback(Callback):

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        print(f"\nBatch #{batch} Time: {batch_time:.2f}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\nEpoch #{epoch} Time: {epoch_time:.2f}")


def main():
    pd.options.mode.chained_assignment = None

    config = load_config(CONFIG_PATH)

    image_h, image_w = config["image_height"], config["image_width"]
    input_h, input_w, input_c = config["input_height"], config["input_width"], config["input_channels"]

    # Load data
    train_df, val_df = load_data(config["train_dataset_path"],
                                 config["train_images_path"],
                                 input_shape=(image_h, image_w),
                                 output_shape=(input_h, input_w),
                                 data_size=config["data_size"],
                                 batch_size=config["batch_size"],
                                 train_size=config["train_size"])

    # print(f"Class weights: {train_df.class_weights}")

    # Init model
    model = init_model((input_h, input_w, input_c))

    # Train the model
    SaveCallback = tf.keras.callbacks.ModelCheckpoint(filepath=config["save_model_path"], monitor='val_loss',
                                                      save_best_only=True, verbose=1)
    history = model.fit(train_df,
                        validation_data=val_df,
                        epochs=config["epoch_num"],
                        callbacks=[DisplayCallback(), SaveCallback])

    # Save history
    save_train_history(history, config["save_history_path"])

    # Visualize training process
    visualize_history(history.history)


try:
    main()
except Exception as e:
    print(f"Error: {e}")
