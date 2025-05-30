
# English Accent Classifier

This repository contains a practical tool to classify English accents from a speaker's voice extracted from a public video URL. It was developed as part of a technical challenge to demonstrate applied AI and deployment capabilities.

## Live Demo

Access the working application through this Hugging Face Space:  
https://huggingface.co/spaces/Stevenf2003/accent-detector

## Objective

The tool takes a public video link (e.g., YouTube or Google Drive), extracts the audio, and predicts the speaker's English accent using a pretrained model. It helps evaluate accent clarity and regional speech characteristics, useful in HR and automation pipelines.

## Features

- Accepts public video URLs (YouTube(did not finish it because we need to bypass bot-detection) or Google Drive)
- Downloads and extracts the first 10 seconds of audio
- Preprocesses audio (mono, 16kHz)
- Classifies the English accent using a fine-tuned Wav2Vec2 model
- Outputs:
  - Detected accent label
  - Confidence score in percentage
  - Explanation string with both details

## Model

- Model: `Jzuluaga/accent-id-commonaccent_xlsr-en-english` (Hugging Face)
- Framework: SpeechBrain with Wav2Vec2.0 XLSR backbone
- Training data includes samples from over 20 English accents such as:
  - United States English
  - Indian and South Asian English
  - British (England) English
  - Australian, Canadian, Nigerian, Singaporean, etc.

## Technology Stack

- Python
- Torchaudio
- yt-dlp (for downloading videos)
- Pydub (audio conversion)
- Gradio (frontend interface)
- SpeechBrain (inference API)

## File Structure

├── app.py                      # Main Gradio application
├── custom_interface.py        # Classifier wrapper using SpeechBrain
├── requirements.txt           # Python dependencies
├── hyperparams.yaml           # Model configuration
└── README.md                  # This file


## Setup Instructions

1. Clone the repository
2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
````

3. Run the app locally:

```bash
python app.py
```

## Deployment

This project is deployed on Hugging Face Spaces for easy access and demo testing. It can also be containerized via Docker or run via Streamlit, Flask, or CLI with minor modifications.


## Notes

* The model performs best on accents with sufficient training data (e.g., US, UK, India).
* For edge cases (e.g., Singaporean or neutral English), results may default to the closest match (e.g., US or England).
* All test screenshots are documented in the submission folder with direct google drive links for each tried accent.


