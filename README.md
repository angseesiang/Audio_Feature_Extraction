# 🎵 Audio Feature Extractor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Analysis-orange)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)](#)

This repository contains my training exercise on **audio feature
extraction** using Python and [Librosa](https://librosa.org/).\
It demonstrates how to **load audio files, extract MFCC features, and
generate spectrograms**. The project also includes unit tests to verify
functionality.

------------------------------------------------------------------------

## 📖 Contents

-   `audio_feature_extractor.py` -- Core class for:
    -   Loading audio (`.wav`) files
    -   Extracting **MFCCs** (Mel-Frequency Cepstral Coefficients)
    -   Generating **spectrograms** (Mel or Linear)
-   `test_audio_feature_extractor.py` -- Unit tests with `unittest`
    covering:
    -   Audio loading
    -   MFCC extraction
    -   Spectrogram generation
-   `example.wav` -- Sample audio file for testing and demonstration
-   `requirements.txt` -- Required Python dependencies

------------------------------------------------------------------------

## 🚀 How to Use

### 1. Clone this repository

``` bash
git clone https://github.com/your-username/audio-feature-extractor.git
cd audio-feature-extractor
```

### 2. Create and activate a virtual environment (recommended)

It is best practice to isolate project dependencies in a virtual
environment.

``` bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate    # On Windows
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the feature extractor

``` python
from audio_feature_extractor import AudioFeatureExtractor

# Initialize
extractor = AudioFeatureExtractor(sampling_rate=22050, n_mfcc=13)

# Load audio
audio, sr = extractor.load_audio("example.wav")

# Extract MFCCs
mfccs = extractor.extract_mfccs(audio, sr)

# Generate spectrogram
fig = extractor.generate_spectrogram(audio, title="Example Spectrogram")
fig.show()
```

### 5. Run tests

``` bash
python -m unittest test_audio_feature_extractor.py
```

------------------------------------------------------------------------

## 🛠️ Requirements

See [`requirements.txt`](requirements.txt): - `numpy` - `scipy` -
`matplotlib` - `librosa`

------------------------------------------------------------------------

## 📌 Notes

-   This project was created during my **AI/ML training** to learn how
    to work with audio data.
-   It provides a **modular and testable** way to extract features from
    audio for downstream tasks like speech recognition or music
    analysis.

------------------------------------------------------------------------

## 📜 License

This repository is shared for **educational purposes**. Please credit if
you use it in your work.
