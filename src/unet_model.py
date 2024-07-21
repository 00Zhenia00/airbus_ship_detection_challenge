import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Conv2DTranspose


def conv_block(inputs, num_filters):
    # First Conv segment
    x = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(inputs)

    # Second Conv segment
    x = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)

    return x


def encoder_block(inputs, num_filters):
    # Apply two conv layers
    x = conv_block(inputs, num_filters)

    # Keep Conv output for skip input
    skip_input = x

    # Apply pooling
    x = MaxPool2D((2, 2))(x)

    return skip_input, x


def decoder_block(inputs, skip, num_filters):
    # Perform upsampling
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)

    # Concatenate skip input with x
    x = Concatenate()([x, skip])

    # Apply two conv layers
    x = conv_block(x, num_filters)

    return x


def build_unet(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Expansive path
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Prepare output on last block
    # Sigmoid activation function has been chosen because of binary segmentation
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    # Define model
    model = Model(inputs, outputs, name="UNET")

    return model


def dice_loss(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2, 3))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2, 3))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice


def init_model(input_shape):
    model = build_unet(input_shape)

    # Compile the model
    # Dice loss performs better at class imbalanced problems by design
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=dice_loss,
                  metrics=['accuracy'])

    return model
