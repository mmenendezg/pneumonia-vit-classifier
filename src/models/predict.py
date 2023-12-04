import tensorflow as tf
import PIL
from huggingface_hub import from_pretrained_keras

from visualization.image_processing import (
    get_image_preprocessor,
    get_attention_image,
    get_attention,
)

# Constants
MODEL_CHECKPOINT = "mmenendezg/vit_pneumonia_classifier"
IMG_SIZE = [224, 224]
IMG_CLASSES = ["Normal", "Pneumonia"]
THRESHOLD = 0.65


def make_prediction(image: PIL.Image):
    """
    Make a single prediction using the given model and image.

    Args:
        model_path: The path to the model to load.
        image: The image to predict.
        n_images: The number of images to predict.

    Returns:
        The predictions and attention vectors.
    """
    # Load the model
    model = from_pretrained_keras(MODEL_CHECKPOINT)
    model.compile()
    model_config = model.get_layer("tf_vi_t_model").get_config()
    w_featmap = IMG_SIZE[0] // model_config["patch_size"]
    h_featmap = IMG_SIZE[1] // model_config["patch_size"]

    # Convert images to tensorflow Dataset
    image = image.convert("RGB")
    permutation = lambda image: tf.transpose(image, perm=[2, 0, 1])
    image_preprocessor = get_image_preprocessor()
    image_tf = permutation(image_preprocessor(image))
    image_shape = tf.constant(image).shape
    image_ds = tf.data.Dataset.from_tensors(image_tf).batch(1)

    # Make predictions
    model_output = model.predict(image_ds, verbose=0)

    predictions = model_output[0]
    predictions = [float(prediction) for prediction in predictions]
    predicted_classes = [1 if pred > THRESHOLD else 0 for pred in predictions]

    # Obtain the attention vector
    attentions = get_attention(
        model_output[1],
        1,
        model_config["num_attention_heads"],
        h_featmap,
        w_featmap,
        image_shape[0],
        image_shape[1],
    )

    # Get the attention image
    attention_image = get_attention_image(attentions, image)

    return (attention_image, {IMG_CLASSES[predicted_classes[0]]: predictions[0]})
