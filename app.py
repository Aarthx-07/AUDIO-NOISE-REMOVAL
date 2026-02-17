from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import noisereduce as nr
import os
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
IMAGE_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Smooth waveform plotting
def plot_waveform(audio_path, image_path):
    y, sr = librosa.load(audio_path, mono=True)

    N = len(y)
    t = np.linspace(0, N/sr, N)

    # Downsample for smooth curve
    points = 2000
    step = max(1, N // points)
    t_ds = t[::step]
    y_ds = y[::step]

    # Smooth visually
    y_ds = gaussian_filter1d(y_ds, sigma=1)

    plt.figure(figsize=(10,4))
    plt.plot(t_ds, y_ds, linewidth=1.5)
    plt.title("Waveform (Trigonometric Representation)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

# AI Noise Reduction
def reduce_noise(input_path, output_path):
    y, sr = librosa.load(input_path, mono=True)

    # Automatically estimate quieter parts as noise
    noise_part = y[np.abs(y) < np.percentile(np.abs(y), 25)]

    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_part,
        prop_decrease=0.8,   # slightly reduced strength
        stationary=False
    )

    # Normalize to restore loudness
    reduced = reduced / np.max(np.abs(reduced))

    sf.write(output_path, reduced, sr)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["audio"]
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        cleaned_filename = "cleaned_" + file.filename
        output_path = os.path.join(OUTPUT_FOLDER, cleaned_filename)

        file.save(upload_path)

        uploaded_wave = "uploaded_wave.png"
        cleaned_wave = "cleaned_wave.png"

        # Plot uploaded
        plot_waveform(upload_path, os.path.join(IMAGE_FOLDER, uploaded_wave))

        # AI noise reduction
        reduce_noise(upload_path, output_path)

        # Plot cleaned
        plot_waveform(output_path, os.path.join(IMAGE_FOLDER, cleaned_wave))

        return render_template("index.html",
                               uploaded_audio=file.filename,
                               cleaned_audio=cleaned_filename,
                               uploaded_wave=uploaded_wave,
                               cleaned_wave=cleaned_wave)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<filename>")
def cleaned_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)

   


