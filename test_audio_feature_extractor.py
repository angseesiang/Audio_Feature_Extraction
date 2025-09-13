
import unittest
import numpy as np
import matplotlib.pyplot as plt
from audio_feature_extractor import AudioFeatureExtractor

class TestAudioFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = AudioFeatureExtractor(sampling_rate=22050, n_mfcc=13)
        cls.audio_path = 'audio_files/example.wav'
        cls.audio, cls.sampling_rate = cls.extractor.load_audio(cls.audio_path)

    def test_load_audio(self):
        audio, sr = self.extractor.load_audio(self.audio_path)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, self.extractor.sampling_rate)

    def test_extract_mfccs(self):
        mfccs = self.extractor.extract_mfccs(self.audio, self.sampling_rate)
        self.assertIsInstance(mfccs, np.ndarray)
        self.assertEqual(mfccs.shape[0], self.extractor.n_mfcc)

    def test_generate_spectrogram(self):
        fig = self.extractor.generate_spectrogram(self.audio, self.sampling_rate)
        self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main()
