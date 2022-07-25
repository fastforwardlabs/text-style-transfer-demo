# Exploring Intelligent Writing Assistance

A demonstration of how the NLP task of _text style transfer_ can be applied to enhance the human writing experience using [HuggingFace Transformers](https://huggingface.co/) and [Streamlit](https://streamlit.io/).

> This repo accompanies Cloudera Fast Forward Labs' [blog series](https://blog.fastforwardlabs.com/2022/03/22/an-introduction-to-text-style-transfer.html) in which we explore the task of automatically neutralizing subjectivity bias in free text.

![](/static/images/app_screenshot.png)

The goal of this application is to demonstrate how the NLP task of text style transfer can be applied to enhance the human writing experience. In this sense, we intend to peel back the curtains on how an intelligent writing assistant might function — walking through the logical steps needed to automatically re-style a piece of text (from informal-to-formal **or** subjective-to-neutral) while building up confidence in the model output.

Through the application, we emphasize the imperative for a human-in-the-loop user experience when designing natural language generation systems. We believe text style transfer has the potential to empower writers to better express themselves, but not by blindly generating text. Rather, generative models, in conjunction with interpretability methods, should be combined to help writers understand the nuances of linguistic style and suggest stylistic edits that may improve their writing.

## Project Structure

```
.
├── LICENSE
├── README.md
├── apps
│   ├── app.py
│   ├── app_utils.py
│   ├── data_utils.py
│   └── visualization_utils.py
├── requirements.txt
├── scripts
│   ├── download_models.py
│   ├── install_dependencies.py
│   └── launch_app.py
├── setup.py
├── src
│   ├── __init__.py
│   ├── content_preservation.py
│   ├── style_classification.py
│   ├── style_transfer.py
│   └── transformer_interpretability.py
├── static
│   └── images
│       ├── app_screenshot.png
│       ├── cldr-favicon.ico
│       └── ffllogo2@1x.png
└── tests
    ├── __init__.py
    └── test_style_transfer.py
```

By launching this applied machine learning prototype (AMP) on CML, the following steps will be taken to recreate the project in your workspace:

1. A Python session is run to install all project dependencies and download and cache all HuggingFace models used throughout the application
2. A Streamlit application is deployed to the project

## Launching the Project on CML

This AMP was developed against Python 3.9. There are two ways to launch the project on CML:

1. **From Prototype Catalog** - Navigate to the AMPs tab on a CML workspace, select the "Exploring Intelligent Writing Assistance" tile, click "Launch as Project", click "Configure Project"
2. **As an AMP** - In a CML workspace, click "New Project", add a Project Name, select "AMPs" as the Initial Setup option, copy in this repo URL, click "Create Project", click "Configure Project"
