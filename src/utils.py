import json
import yaml
import matplotlib.pyplot as plt


# Function to load yaml configuration file
def load_config(config_path):
    """ Function to load yaml configuration file. """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def save_train_history(history, save_path):
    """ Save model training history dictionary. """
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(save_path, 'w'))


def load_train_history(save_path):
    """ Load model training history dictionary. """
    return json.load(open(save_path, 'r'))


def visualize_history(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_range = range(len(loss))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Dice Loss')
    plt.plot(epochs_range, val_loss, label='Validation Dice Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Dice Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.legend()

    plt.show()
