"""Wav2Vec2 audio embeddings for overt-production syllable segments."""

import numpy as np
import torch
import torchaudio
import transformers
from transformers import Wav2Vec2Model, Wav2Vec2Processor

transformers.logging.set_verbosity_error()  # Suppress warnings from transformers

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-italian"
# Pinned Hugging Face commit, so that embeddings stay reproducible if the
# checkpoint is updated upstream.
MODEL_REVISION = "e2760fe9ade861421db16f4548f6c0d48568ae0c"
MODEL_SAMPLING_FREQUENCY = 16000


class EmbeddingExtractor:
    """Extract embeddings from audio using a pre-trained, frozen Wav2Vec2 model."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # Resamplers cached by original sample rate
        self._resamplers = {}

    def _check_and_resample(self, waveform: np.ndarray, sample_rate: int):
        """Resample to the model's 16 kHz input rate if necessary."""
        if sample_rate == MODEL_SAMPLING_FREQUENCY:
            return waveform
        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=MODEL_SAMPLING_FREQUENCY,
            )
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        return self._resamplers[sample_rate](waveform_tensor).squeeze(0).numpy()

    def _inputs(self, waveform: np.ndarray, sample_rate: int):
        resampled_waveform = self._check_and_resample(waveform, sample_rate)
        return self.processor(
            resampled_waveform, return_tensors="pt", padding=True,
            sampling_rate=MODEL_SAMPLING_FREQUENCY,
        ).to(self.device)

    def extract_embedding(self, waveform: np.ndarray, sample_rate: int):
        """Extract embeddings from a raw audio waveform.

        The embeddings are hidden_states[0], i.e. the transformer encoder input
        (projected CNN features plus positional embedding), before any
        transformer layer.

        Args:
            waveform (np.ndarray): The raw 1D audio waveform.
            sample_rate (int): The sample rate of the audio waveform.
        Returns:
            np.ndarray: (n_frames, 1024) embedding sequence.
        """
        inputs = self._inputs(waveform, sample_rate)
        # Compute only up to hidden_states[0], skipping the transformer layers.
        # Mirrors Wav2Vec2Model/Wav2Vec2Encoder(StableLayerNorm).forward in eval
        # mode for a single unpadded waveform (no attention mask, no SpecAugment).
        with torch.no_grad():
            features = self.model.feature_extractor(inputs.input_values).transpose(1, 2)
            hidden_states, _ = self.model.feature_projection(features)
            encoder = self.model.encoder
            hidden_states = hidden_states + encoder.pos_conv_embed(hidden_states)
            if not self.model.config.do_stable_layer_norm:
                hidden_states = encoder.layer_norm(hidden_states)
        return hidden_states.squeeze(0).cpu().numpy()

    def extract_contextual_embedding(self, waveform: np.ndarray, sample_rate: int):
        """Extract contextual embeddings from a raw audio waveform.

        The embeddings are the average of the last four transformer layers.

        Args:
            waveform (np.ndarray): The raw 1D audio waveform.
            sample_rate (int): The sample rate of the audio waveform.
        Returns:
            np.ndarray: (n_frames, 1024) contextual embedding sequence.
        """
        inputs = self._inputs(waveform, sample_rate)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0).squeeze(0).cpu().numpy()
