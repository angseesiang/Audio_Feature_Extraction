# audio_feature_extractor.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class AudioFeatureExtractor:
    """
    Simple audio feature extractor built on top of librosa.

    Parameters
    ----------
    sampling_rate : int
        Target sampling rate used for loading audio and computing features.
    n_mfcc : int
        Number of MFCC coefficients to compute.
    hop_length : int
        Hop length for STFT-related features.
    n_fft : int
        FFT window size for STFT-related features.
    """

    def __init__(
        self,
        sampling_rate: int = 22_050,
        n_mfcc: int = 13,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

    # ---------------------------
    # Loading
    # ---------------------------
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file as a mono waveform at the configured sampling rate.

        Returns
        -------
        audio : np.ndarray
            1D numpy array of shape (n_samples,)
        sr : int
            Sampling rate actually used (== self.sampling_rate)
        """
        p = Path(file_path)

        # If exact path doesn't exist, try a case-insensitive fallback with same stem
        if not p.exists():
            alt = None
            if p.parent.exists():
                for q in p.parent.iterdir():
                    if q.is_file() and q.stem.lower() == p.stem.lower():
                        alt = q
                        break
            if alt is None:
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            p = alt

        try:
            # librosa.load defaults: mono=True (mixes down), dtype=float32
            audio, sr = librosa.load(
                str(p),
                sr=self.sampling_rate,
                mono=True,
            )
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio '{p}': {e}") from e

    # ---------------------------
    # Features
    # ---------------------------
    def extract_mfccs(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Compute MFCCs from a mono waveform.

        Parameters
        ----------
        audio : np.ndarray
            1D waveform (mono)
        sr : int, optional
            Sampling rate to use (defaults to self.sampling_rate).

        Returns
        -------
        mfccs : np.ndarray
            2D array of shape (n_mfcc, n_frames)
        """
        if audio.ndim != 1:
            raise ValueError("Expected mono waveform (1D array).")

        sr = sr or self.sampling_rate

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        return mfccs

    # ---------------------------
    # Visualization
    # ---------------------------
    def generate_spectrogram(
        self,
        audio: np.ndarray,
        title: str = "Spectrogram",
        use_mel: bool = True,
    ) -> plt.Figure:
        """
        Generate and return a spectrogram figure for the given waveform.

        Parameters
        ----------
        audio : np.ndarray
            1D waveform (mono)
        title : str
            Title for the figure.
        use_mel : bool
            If True, plot a mel-spectrogram in dB. Otherwise, plot linear
            magnitude spectrogram in dB.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created matplotlib figure (useful for unit tests).
        """
        if audio.ndim != 1:
            raise ValueError("Expected mono waveform (1D array).")

        if use_mel:
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,  # power spectrogram
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            data = S_db
            y_axis = "mel"
        else:
            # Linear-frequency magnitude spectrogram (in dB)
            S = np.abs(
                librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length)
            )
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            data = S_db
            y_axis = "log"  # log-frequency axis for readability

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        img = librosa.display.specshow(
            data,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis=y_axis,
            ax=ax,
        )
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()

        return fig

