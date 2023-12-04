from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

# Constants
IMG_SIZE = [224, 224]


def get_image_preprocessor() -> Sequential:
    """
    Creates a Sequential model that preprocesses an image for the CNN.

    The model consists of two layers:
    * Resizing: Resizes the image to the desired size (IMG_SIZE).
    * Rescaling: Rescales the image pixels to be between 0 and 1.

    Args:
        None

    Returns:
        Sequential: The image preprocessor model.
    """
    image_preprocessor = Sequential(
        [
            layers.Resizing(
                height=IMG_SIZE[0],
                width=IMG_SIZE[1],
                interpolation="nearest",
            ),
            layers.Rescaling(scale=1.0 / 255.0),
        ]
    )
    return image_preprocessor


def get_attention_image(attentions: tf.Tensor, image: Image) -> Image:
    """
    Creates an attention image from a tensor of attention weights and an image.

    The attention image is created by multiplying the attention weights with the corresponding pixels in the original image.

    Args:
        attentions (tf.Tensor): A tensor of attention weights.
        image (Image): The original image.

    Returns:
        Image: The attention image.
    """
    rescaled_image = tf.cast(image, dtype=tf.float32) / 255.0
    attention_image = rescaled_image * attentions[0]
    attention_image = tf.image.rgb_to_grayscale(attention_image)
    attention_image = tf.squeeze(attention_image, axis=-1)
    attention_image = plt.cm.viridis(attention_image)
    attention_image = Image.fromarray(np.uint8(attention_image * 255))
    return attention_image


def process_attention(
    attention_vals: tf.Tensor,
    n_heads: int,
    h_featmap: int,
    w_featmap: int,
    h_original: int,
    w_original: int,
):
    """
    Process the attention weights for visualization.

    Args:
        attention_vals: The attention weights.
        n_heads: The number of attention heads.
        h_featmap: The height of the feature map.
        w_featmap: The width of the feature map.
        h_original: The height of the original image.
        w_original: The width of the original image.

    Returns:
        The processed attention weights.
    """
    # We only keep the output patch attention
    attention = tf.expand_dims(attention_vals, axis=0)
    attention = tf.reshape(attention[0, :, 0, 1:], (n_heads, w_featmap, h_featmap, 1))
    # Aggregation of the n heads in the last layer
    attention = tf.reduce_mean(attention, axis=0)
    attention = tf.image.resize(
        attention, size=[h_original, w_original], method="lanczos5"
    )
    # Normalize the attention values to have values from zero to one
    attention -= tf.reduce_min(attention)
    attention /= tf.reduce_max(attention)
    return attention


def get_attention(
    attentions: tf.Tensor,
    examples: int,
    num_attention_heads: int,
    h_featmap: int,
    w_featmap: int,
    h_original: int,
    w_original: int,
):
    """
    Get the attention weights for a batch of images.

    Args:
        attentions: The attention weights.
        examples: The number of examples.
        num_attention_heads: The number of attention heads.
        h_featmap: The height of the feature map.
        w_featmap: The width of the feature map.
        h_original: The height of the original image.
        w_original: The width of the original image.

    Returns:
        The attention weights.
    """
    attentions = tf.reshape(attentions, (examples, num_attention_heads, -1))
    last_dimension = int(tf.math.sqrt(float(attentions.shape[-1])).numpy())
    attentions = tf.reshape(
        attentions, (examples, num_attention_heads, last_dimension, last_dimension)
    )

    attention_list = []
    for attention in attentions:
        processed_attenttion = process_attention(
            attention, num_attention_heads, h_featmap, w_featmap, h_original, w_original
        )
        attention_list.append(processed_attenttion)
    return attention_list
