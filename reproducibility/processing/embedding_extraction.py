import torch
import torchaudio
import transformers
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
transformers.logging.set_verbosity_error()  # Suppress warnings from transformers

class EmbeddingExtractor:
    """Class to extract embeddings from audio using a pre-trained Wav2Vec2 model."""
    def __init__(self):
        # model_name = "facebook/wav2vec2-base-960h"
        model_name = "facebook/wav2vec2-large-xlsr-53-italian"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Resamplers cached by original sample rate
        self._resamplers = {}

    def _check_and_resample(self, waveform: np.ndarray, sample_rate: int):
        """Check the sample rate and resample if necessary."""
        if sample_rate != 16000:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            # Create a tensor for resampling, then convert back to numpy array
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            resampled_waveform = self._resamplers[sample_rate](waveform_tensor).squeeze(0).numpy()
            return resampled_waveform
        return waveform
    
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int):
        """Extract embeddings from a raw audio waveform.
            The embeddings are taken from hidden_states[0], i.e. the transformer encoder
            input (projected CNN features plus positional embedding), before any
            transformer layer.
            Args:
                waveform (np.ndarray): The raw audio waveform.
                sample_rate (int): The sample rate of the audio waveform.
            Returns:
                np.ndarray: The extracted embeddings.
        """
        # The processor expects a raw 1D waveform or a list of them.
        # It will handle batching and tensor conversion.
        
        resampled_waveform = self._check_and_resample(waveform, sample_rate)            
        inputs = self.processor(resampled_waveform, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
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

        # Return the transformer encoder input (before the first transformer layer)
        return hidden_states.squeeze(0).cpu().numpy() # [n_frames, 1024]
    
        # with torch.no_grad():
        #     outputs = self.model.feature_extractor(inputs.input_values)

        # return outputs.squeeze(0).transpose(0, 1).cpu().numpy() # [24, 512]
    
    def extract_contextual_embedding(self, waveform: np.ndarray, sample_rate: int):
        """Extract contextual embeddings from a raw audio waveform.
            The embeddings are taken from the last transformer layer.
            Args:
                waveform (np.ndarray): The raw audio waveform.
                sample_rate (int): The sample rate of the audio waveform.
            Returns:
                np.ndarray: The extracted contextual embeddings.
        """

        resampled_waveform = self._check_and_resample(waveform, sample_rate)            
        inputs = self.processor(resampled_waveform, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Return the average over last 4 transformer layers as contextual embedding
        return torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0).squeeze(0).cpu().numpy()
