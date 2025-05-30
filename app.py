import os
import uuid
import gradio as gr
import torchaudio
import torch
import yt_dlp
from pydub import AudioSegment
from custom_interface import CustomEncoderWav2vec2Classifier

# Load the model correctly using from_hparams (NO foreign_class)
classifier = CustomEncoderWav2vec2Classifier.from_hparams(
    source=".",
    hparams_file="hyperparams.yaml",
    savedir=".",
)

def process_video(url):
    try:
        uid = str(uuid.uuid4())
        video_path = f"/tmp/{uid}.mp4"
        audio_path = f"/tmp/{uid}.wav"

        if "drive.google.com" in url:
            url = url.replace("/view?usp=sharing", "").replace("open?", "uc?export=download&")

        ydl_opts = {
            'outtmpl': video_path,
            'format': 'mp4',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Extract & trim first 30 seconds audio
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio = audio[:10_000]  # trim to 30s max
        audio.export(audio_path, format="wav")

        # Load waveform
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = transform(waveform)

        # Classify
        out_prob, score, index, text_lab = classifier.classify_batch(waveform)
        return text_lab[0], f"{float(score[0]) * 100:.2f}%", f"The accent is likely {text_lab[0]} with {float(score[0]) * 100:.2f}% confidence."

    except Exception as e:
        return " Failed to process video", "", str(e)

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🎙️ English Accent Classifier\nPaste a public video URL (YouTube, Google Drive, etc.)")
    with gr.Row():
        url = gr.Textbox(label="Public Video URL")
        btn = gr.Button("Analyze")
    out1 = gr.Textbox(label="Detected Accent")
    out2 = gr.Textbox(label="Confidence")
    out3 = gr.Textbox(label="Explanation")

    btn.click(fn=process_video, inputs=url, outputs=[out1, out2, out3])

app.launch()
