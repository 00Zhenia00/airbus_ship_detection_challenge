# Airbus Ship Detection Challenge

# 1. About

This project is solution of Airbus Ship Detection Challenge. It is used deep learning approach, specifically U-Net neural network is built. The Dice score is used as a metric. For model creation python Tensorflow Keras is used.

# 2. How to use

To install all required dependencies run:

```python
> pip install -r requirements.txt
```

The `config.yaml` file is used to access or change the training parameters of the deep learning model. In config file you should specify train and test data paths, paths for model and training history saving, and U-Net model parameters.

To train model with given parameters run the following command:

```python
> python train_model.py
```

While training the most efficient model is saved in `./models/best_weights.keras` directory.

For model inference test run:

```python
> python predict_model.py
```

# 3. Description

## 3.1. Exploratory analysis

To review exploratory analysis please find jupyter notebook `./notebooks/eda.ipynb`.

## 3.2. Data loading

Data loading is performed by `load_data(…)` function (`load_data.py`).

First of all, training dataset is grouped by `ImageId`attribute and all masks in rle-format are added to list.

Because the data contains training and testing data *only*, and because it is needed to split in on  train, validation and test datasets, we will split training dataset on train (80%) and validation (20%) parts.

Based on the results of exploratory analysis, the dataset is unbalanced (it is much more non-ship images than ship images). 

First, class weights were assigned to target predictions of ship and non-ship images (relative frequencies of ship and non-ship images) to put more importance on the samples that are underrepresented (ship images). But this approach needed a lot of samples and resources to become efficient enough.

So, second time it was used next dataset balancing: ship and non-ship images were taken with given ratios. In this case, 75% ship images and 25% non-ship images were taken to balanced dataset.

To represent train and validation datasets it is used `AirbusDataset` class that inherits `tensorflow.keras.utils.Sequence` class. When obtaining AirbusDataset container item (`AirbusDataset .__getitem__(self, index)`), it returns batch of defined batch size that includes images, masks (, and corresponding weights if applied). On each epoch dataset samples are shuffled.

AirbusDataset output samples are also preprocessed: images are normalized, images and masks are resized to model-required shape. Performing image normalization simply involves casting image Tensors to `float32` format and division by `255.0`. Images/masks are resized from `(768, 768, 3)`/`(768, 768, 1)` to `(128, 128, 3)`/`(128, 128, 1)` size to to simplify the calculations. Images are passed to U-Net model in normalized and resized format.

## 3.3. Model architecture

The number of feature maps generated at the first U-net convolutional block will be 64. In total, neural network will consist of 5 U-Net blocks and will have 2 feature maps in the *final 1x1 Conv layer*.

The shape of  input image will be (128, 128, 3) - RGB image.

Adam optimizer is used in the model.

As the main metric dice loss is used. Dice loss performs better at class imbalanced problems by design. Also, accuracy is used as an additional metric.

Model is composed of a contracting path, which is built from encoder blocks, and an expansive path built from decoder blocks. At each individual level the output of a encoder block is connected to an decoder block with skip connection.

Encoder block is composed of two 3x3 convolutional blocks with ReLU activated outputs, which are followed by Max pooling block with a stride of 2, so that the output can be used by the next convolutional block.

Decoder block is composed of transposed convolution layer Conv2DTranspose for data upsampling, a concatenation with the corresponding skip connection data, and two 3x3 convolutions, each followed by a ReLU.

Finally, a 1x1 convolution layer, followed by sigmoid activation function (because of binary classification) is used to reduce the number of channels to the desired number of classes.

`build_unet` function is used to build the model (`unet_model.py`).

`init_model` function is used to compile model with given optimizer, loss function, and additional metrics (`unet_model.py`).

## 3.4. Model training

The model was trained on a sample size of 3000 from a training dataset for 10 epochs. The batch size was 32.

Also `ModelCheckpoint`callback was included to save the model to disk after each epoch. It was configured that it only saves highest performance model.

## 3.5. Model inference

To see model inference use `predict_model.py` script. It loads saved trained model and uses it to predict masks of random 5 samples from the test dataset.

## 3.6. Results

Model training result and inference test are located in `./reports` directory.

## 3.7. Improvements

Possible improvements:
s
- training on bigger sample and more epochs
- using modificated Dice loss function, for example, BCE-Dice loss function
- Add batch normalization layers in model