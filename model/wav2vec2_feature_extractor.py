import librosa
import torch
from torch import nn
from transformers import AutoProcessor, Wav2Vec2Model


class Wav2Vec2ForFeatureExtraction(nn.Module):
    def __init__(self):
        super(Wav2Vec2ForFeatureExtraction, self).__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        path=None,
        start_frame=None,
        sound_vector=None,
        fps=25,
        num_frame=750,
        device="cpu",
    ):
        
        if sound_vector is None:
            sound_vector = self.read_audio(
                path, start_frame, fps=fps, num_frame=num_frame, device=device
            )

        inputs = self.processor(sound_vector, sampling_rate=16000, return_tensors="pt")

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs["extract_features"]

        return features
    

    def read_audio(self, path, start_frame, fps=25, num_frame=750, device="cpu"):
        sampling_rate = 16000  # wav2vec uses 16k sampling rate
        duration = num_frame / fps
        offset = start_frame / fps
        y, s = librosa.load(
            path, offset=offset, duration=duration, sr=sampling_rate, mono=True
        )
        y = torch.from_numpy(y)
        y = y.to(device)
        return y
