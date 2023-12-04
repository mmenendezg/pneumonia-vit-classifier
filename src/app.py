import os
import gradio as gr
from models.predict import make_prediction

# Badges
KAGGLE_NOTEBOOK = "[![Static Badge](https://img.shields.io/badge/Open_Notebook_in_Kaggle-gray?logo=kaggle&logoColor=white&labelColor=20BEFF)](https://www.kaggle.com/code/mmenendezg/pneumonia-classifier-using-vit)"
GITHUB_REPOSITORY = "[![Static Badge](https://img.shields.io/badge/Git_Repository-gray?logo=github&logoColor=white&labelColor=181717)](https://github.com/mmenendezg/pneumonia_x_ray)"
HF_SPACE = "[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/mmenendezg/pneumonia_vit_classifier)"

# Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown(
    f"""
    # Pneumonia Classifier

    This is a space to test the Pneumonia Classifier model.

    {KAGGLE_NOTEBOOK}

    {GITHUB_REPOSITORY}

    {HF_SPACE}
    """
    )
    with gr.Row():
        with gr.Column():
            uploaded_image = gr.Image(
                label="Chest X-ray image",
                sources=["upload", "clipboard"],
                type="pil",
                height=550,
            )
        with gr.Column():
            labels = gr.Label(label="Prediction")
            attention_image = gr.Image(
                label="Attention zones", image_mode="L", height=425
            )
    with gr.Row():
        classify_btn = gr.Button("Classify", variant="primary")
        clear_btn = gr.ClearButton(components=[uploaded_image, labels, attention_image])
    classify_btn.click(
        fn=make_prediction, inputs=uploaded_image, outputs=[attention_image, labels]
    )
    gr.Examples(
        examples=[
            os.path.join(os.path.dirname(__file__), "examples/normal.jpeg"),
            os.path.join(os.path.dirname(__file__), "examples/pneumonia.jpeg"),
        ],
        inputs=uploaded_image,
    )
demo.launch()
