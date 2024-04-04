# https://arxiv.org/pdf/1910.05453.pdf

import os

import fairseq
import librosa
import torch
import torch.nn as nn


class SoundTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model_path = self.get_checkpoint()
        (
            self.model,
            self.cfg,
            self.task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = self.model[0]

        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def get_checkpoint(self):
        file_name = "vq-wav2vec.pt"

        if not os.path.exists(file_name):
            import requests

            url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt"

            print("Downloading vq-wav2vec.pt")
            r = requests.get(url)
            open(file_name, "wb").write(r.content)
            print("Downloaded vq-wav2vec.pt")

        return file_name

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

    def forward(self, sound):
        z = self.model.feature_extractor(sound)
        _, idx = self.model.vector_quantizer.forward_idx(z)
        return idx


if __name__ == "__main__":
    wav_input = torch.randn(1, 10000)

    tokenizer = SoundTokenizer()

    print(tokenizer(wav_input))

    #
