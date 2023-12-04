# Pneumonia Classifier 

This is a project to fine-tune a pretrained `ViT` to identify pneumonia in chest x-ray images.

The goal of this project is to take the `Pneumonia Chest X-Ray` dataset, upload it to the [`Hugging Face Hub`](https://huggingface.co/datasets/mmenendezg/pneumonia_x_ray), create a classifier to detect pneumonia, and finally upload the model to the Hugging Face Hub. 

[![Static Badge](https://img.shields.io/badge/Open_Notebook_in_Kaggle-gray?logo=kaggle&logoColor=white&labelColor=20BEFF)](https://www.kaggle.com/code/mmenendezg/pneumonia-classifier-using-vit)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/mmenendezg/pneumonia_vit_classifier)

Directory Structure
--------------------

    .
    ├── README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
    ├── config  <- any configuration files
    ├── data
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   ├── figures 
    │   └── raw <- original unmodified data acting as source of truth and 
    └── pneumonia_vit_classifier
        ├── examples <- images used as examples in the HuggingFace Space
        ├── models  <- scripts for the prediction of the images
        ├── visualization  <- scripts to visualize the atttention vectors
        ├── app.py  <- script to execute the gradio app
        └── requirements.txt  <- text file containing the required libraries
