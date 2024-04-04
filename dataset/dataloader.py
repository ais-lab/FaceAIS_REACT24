import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")
from functools import cmp_to_key

import lightning as pl

test_video_list = [
    "NoXI/001_2016-03-17_Paris/Expert_video/21",
    "NoXI/023_2016-04-25_Paris/Expert_video/25",
    "NoXI/019_2016-04-20_Paris/Expert_video/13",
    "RECOLA/group-2/P41/2",
]


def is_submit_video(video_path):
    for test_video in test_video_list:
        if test_video in video_path:
            print(f"there is {video_path}")
            return True
    return False


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        img = transform(img)
        return img


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames


def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[
            i * frame_n_samples : shifted_n_samples
            + i * frame_n_samples
            + frame_n_samples
        ]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(
            torch.from_numpy(curr_samples).float().view(1, -1),
            sample_frequency=sr,
            use_energy=True,
        )
        curr_mfcc = curr_mfcc.transpose(0, 1)  # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack(
            (curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())
        ).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(
        self,
        root_path,
        split,
        img_size=256,
        crop_size=224,
        clip_length=751,
        fps=25,
        load_audio=True,
        load_video_s=True,
        load_video_l=True,
        load_emotion_s=False,
        load_emotion_l=False,
        load_3dmm_s=False,
        load_3dmm_l=False,
        load_ref=True,
        repeat_mirrored=True,
        load_raw_audio=False,
        submit_video_only=False,
        is_load_video_address=False,
    ):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self._fps = fps
        self._split = split

        self.submit_video_only = submit_video_only
        self.is_load_video_address = is_load_video_address

        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(
            os.path.join(self._root_path, self._split + ".csv"),
            header=None,
            delimiter=",",
        )
        self._list_path = self._list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        self.load_raw_audio = load_raw_audio

        self._audio_path = os.path.join(self._data_path, "Audio_files")
        self._video_path = os.path.join(self._data_path, "Video_files")
        self._emotion_path = os.path.join(self._data_path, "Emotion")
        self._3dmm_path = os.path.join(self._data_path, "3D_FV_files")

        self.mean_face = torch.FloatTensor(
            np.load("external/FaceVerse/mean_face.npy").astype(np.float32)
        ).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load("external/FaceVerse/std_face.npy").astype(np.float32)
        ).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])

        if (
            self._split in ["val", "test"] or repeat_mirrored
        ):  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

        self.data_list = []
        for i, (sp, lp) in enumerate(zip(speaker_path, listener_path)):
            if self.submit_video_only and not is_submit_video(sp):
                continue

            ab_speaker_video_path = os.path.join(self._video_path, sp)
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + ".wav")
            ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + ".csv")
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + ".npy")

            ab_listener_video_path = os.path.join(self._video_path, lp)
            ab_listener_audio_path = os.path.join(self._audio_path, lp + ".wav")
            ab_listener_emotion_path = os.path.join(self._emotion_path, lp + ".csv")
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + ".npy")

            self.data_list.append(
                {
                    "speaker_video_path": ab_speaker_video_path,
                    "speaker_audio_path": ab_speaker_audio_path,
                    "speaker_emotion_path": ab_speaker_emotion_path,
                    "speaker_3dmm_path": ab_speaker_3dmm_path,
                    "listener_video_path": ab_listener_video_path,
                    "listener_audio_path": ab_listener_audio_path,
                    "listener_emotion_path": ab_listener_emotion_path,
                    "listener_3dmm_path": ab_listener_3dmm_path,
                }
            )

        self._len = len(self.data_list)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        # if self._split == "train":  # only done at training time
        #     changed_sign = random.randint(0, 1)

        speaker_prefix = "speaker" if changed_sign == 0 else "listener"
        listener_prefix = "listener" if changed_sign == 0 else "speaker"

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f"{speaker_prefix}_video_path"]
        listener_video_path = data[f"{listener_prefix}_video_path"]

        img_paths = os.listdir(speaker_video_path)
        total_length = len(img_paths)
        img_paths = sorted(
            img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4]))
        )
        cp = (
            random.randint(0, total_length - 1 - self._clip_length)
            if self._clip_length < total_length
            else 0
        )  if self._split == "train" else 0

        img_paths = img_paths[cp : cp + self._clip_length]

        speaker_video_clip = 0
        speaker_video_address = []
        if self.load_video_s:
            clip = []
            for img_path in img_paths:
                img_address = os.path.join(speaker_video_path, img_path)
                speaker_video_address.append(img_address)

                if self.is_load_video_address:
                    continue

                img = self._img_loader(img_address)
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            if not self.is_load_video_address:
                speaker_video_clip = torch.cat(clip, dim=0)

        # listener video clip only needed for val/test
        listener_video_clip = 0
        listener_video_address = []
        if self.load_video_l:
            clip = []
            for img_path in img_paths:
                img_address = os.path.join(listener_video_path, img_path)
                listener_video_address.append(img_address)

                if self.is_load_video_address:
                    continue

                img = self._img_loader(img_address)
                img = self._transform(img)
                clip.append(img.unsqueeze(0))

            if not self.is_load_video_address:
                listener_video_clip = torch.cat(clip, dim=0)

        if self.is_load_video_address:
            speaker_video_clip = speaker_video_address
            listener_video_clip = listener_video_address

        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f"{speaker_prefix}_audio_path"]

            if self.load_raw_audio:
                # return the path only
                speaker_audio_clip = (speaker_audio_path, cp)
            else:
                # # no-need to process on the fly, just preprocess it once
                # pre_extracted_audio_path = speaker_audio_path.replace(".wav", ".npy")
                # # check for pre-extracted audio
                # if os.path.exists(pre_extracted_audio_path):
                #     speaker_audio_clip = np.load(pre_extracted_audio_path)

                # else:
                speaker_audio_clip = extract_audio_features(
                    speaker_audio_path, self._fps, total_length
                )
                # save extracted audio
                # np.save(pre_extracted_audio_path, speaker_audio_clip)

                speaker_audio_clip = speaker_audio_clip[cp : cp + self._clip_length]

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        if self.load_emotion_l:
            listener_emotion_path = data[f"{listener_prefix}_emotion_path"]
            listener_emotion = pd.read_csv(
                listener_emotion_path, header=None, delimiter=","
            )
            listener_emotion = torch.from_numpy(
                np.array(listener_emotion.drop(0)).astype(np.float32)
            )[cp : cp + self._clip_length]

        if self.load_emotion_s:
            speaker_emotion_path = data[f"{speaker_prefix}_emotion_path"]
            speaker_emotion = pd.read_csv(
                speaker_emotion_path, header=None, delimiter=","
            )
            speaker_emotion = torch.from_numpy(
                np.array(speaker_emotion.drop(0)).astype(np.float32)
            )[cp : cp + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f"{listener_prefix}_3dmm_path"]
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp : cp + self._clip_length]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f"{speaker_prefix}_3dmm_path"]
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp : cp + self._clip_length]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            img_paths = os.listdir(listener_video_path)
            img_paths = sorted(
                img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4]))
            )
            listener_reference = self._img_loader(
                os.path.join(listener_video_path, img_paths[0])
            )
            listener_reference = self._transform(listener_reference)

        return (
            speaker_video_clip,
            speaker_audio_clip,
            speaker_emotion,
            speaker_3dmm,
            listener_video_clip,
            listener_audio_clip,
            listener_emotion,
            listener_3dmm,
            listener_reference,
        )

    def __len__(self):
        return self._len


@dataclass
class Dataconfig:
    dataset_path: str = None
    batch_size: int = 1
    num_workers: int = 12
    img_size: int = 256
    crop_size: int = 224
    clip_length: int = 736
    test_extend_factor: int = 1
    is_render: bool = False
    submit_video_only: bool = False


class ReactDataModule(pl.LightningDataModule):
    def __init__(
        self,
        conf,
        load_audio=True,
        load_video=True,
        load_emotion=True,
        load_3dmm=True,
        load_ref=True,
        load_raw_audio=False,
        load_video_address=False,
        only_speaker=False,
        collect_metrics_in="test",
    ):
        super().__init__()
        self.conf = conf
        self.load_audio = load_audio
        self.load_video = load_video
        self.load_emotion = load_emotion
        self.load_3dmm = load_3dmm
        self.load_ref = load_ref
        self.load_raw_audio = load_raw_audio
        self.load_video_address = load_video_address
        self.only_speaker = only_speaker
        self.collect_metrics_in = collect_metrics_in

    def setup(self, stage):
        # train_loader = get_dataloader(args, "train", load_audio=True, load_video_s=True,  load_emotion_l=True,  load_3dmm_l=True)
        # val_loader = get_dataloader(args, "val", load_audio=True, load_video_s=True,  load_emotion_l=True, load_3dmm_l=True, load_ref=True)

        self.train_ds = ReactionDataset(
            self.conf.dataset_path,
            "train",
            img_size=self.conf.img_size,
            crop_size=self.conf.crop_size,
            clip_length=self.conf.clip_length,
            load_audio=self.load_audio,
            load_video_s=self.load_video,
            load_video_l=self.load_video if not self.only_speaker else False,
            load_emotion_s=self.load_emotion,
            load_emotion_l=self.load_emotion if not self.only_speaker else False,
            load_3dmm_s=self.load_3dmm,
            load_3dmm_l=self.load_3dmm if not self.only_speaker else False,
            load_ref=self.load_ref,
            repeat_mirrored=True,
            load_raw_audio=self.load_raw_audio,
            is_load_video_address=self.load_video_address,
        )

        self.val_ds = ReactionDataset(
            self.conf.dataset_path,
            "val",
            img_size=self.conf.img_size,
            crop_size=self.conf.crop_size,
            clip_length=self.conf.clip_length,
            load_audio=self.load_audio,
            load_video_s=self.load_video,
            load_video_l=self.load_video if not self.only_speaker else False,
            load_emotion_s=self.load_emotion,
            load_emotion_l=self.load_emotion if not self.only_speaker else False,
            load_3dmm_s=self.load_3dmm,
            load_3dmm_l=self.load_3dmm if not self.only_speaker else False,
            load_ref=self.load_ref,
            repeat_mirrored=True,
            load_raw_audio=self.load_raw_audio,
            submit_video_only=self.conf.submit_video_only,
            is_load_video_address=self.load_video_address,
        )

        self.test_ds = ReactionDataset(
            self.conf.dataset_path,
            self.collect_metrics_in,
            img_size=self.conf.img_size,
            crop_size=self.conf.crop_size,
            clip_length=self.conf.clip_length,
            load_audio=self.load_audio,
            load_video_s=self.load_video,
            load_video_l=self.load_video if not self.only_speaker else False,
            load_emotion_s=self.load_emotion,
            load_emotion_l=self.load_emotion if not self.only_speaker else False,
            load_3dmm_s=self.load_3dmm,
            load_3dmm_l=self.load_3dmm if not self.only_speaker else False,
            load_ref=self.load_ref,
            repeat_mirrored=True,
            load_raw_audio=self.load_raw_audio,
            submit_video_only=self.conf.submit_video_only,
            is_load_video_address=self.load_video_address,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        # return as a list for testing with diversity
        return [
            DataLoader(
                dataset=self.test_ds,
                batch_size=self.conf.batch_size,
                shuffle=False,
                num_workers=self.conf.num_workers,
                # pin_memory=True,
                # persistent_workers=True,
            )
            for i in range(self.conf.test_extend_factor)
        ]

    # def test_dataloader(self):
    #     return DataLoader(dataset=self.test_ds, batch_size=self.conf.batch_size, shuffle=False, num_workers=self.conf.num_workers)


def get_dataloader(
    conf,
    split,
    load_audio=False,
    load_video_s=False,
    load_video_l=False,
    load_emotion_s=False,
    load_emotion_l=False,
    load_3dmm_s=False,
    load_3dmm_l=False,
    load_ref=False,
    repeat_mirrored=True,
    load_raw_audio=False,
):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    # print('==> Preparing data for {}...'.format(split) + '\n')
    dataset = ReactionDataset(
        conf.dataset_path,
        split,
        img_size=conf.img_size,
        crop_size=conf.crop_size,
        clip_length=conf.clip_length,
        load_audio=load_audio,
        load_video_s=load_video_s,
        load_video_l=load_video_l,
        load_emotion_s=load_emotion_s,
        load_emotion_l=load_emotion_l,
        load_3dmm_s=load_3dmm_s,
        load_3dmm_l=load_3dmm_l,
        load_ref=load_ref,
        repeat_mirrored=repeat_mirrored,
        load_raw_audio=load_raw_audio,
    )
    shuffle = True if split == "train" else False
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
    )
    return dataloader


def divide_seq_and_pad(
    original_tensor: torch.Tensor, max_lenth: int, padding_value: int = -100, factor=1
):
    b, t = original_tensor.size()  # batch size, sequence length

    # Calculate the new shape
    # new_b = b * int(math.ceil(t / new_t))

    # iterate through each seq in batch, divide it into new_t chunks
    final_tensor = []

    max_lenth = max_lenth * factor

    for i in range(b):
        seq = original_tensor[i]
        seq = seq.unsqueeze(0)
        seq = torch.split(seq, max_lenth, dim=1)

        # cast tuple to list
        seq = list(seq)

        # pad the last seq if it is not new_t
        if seq[-1].size(1) != max_lenth:
            seq[-1] = F.pad(
                seq[-1],
                (max_lenth - seq[-1].size(1), 0),
                mode="constant",
                value=padding_value,
            )

        seq = torch.cat(seq, dim=0)
        final_tensor.append(seq)

    final_tensor = torch.cat(final_tensor, dim=0)

    pad_mask = (
        final_tensor != padding_value
    )  # True indicates that the element should take part in attention.

    return final_tensor, pad_mask


if __name__ == "__main__":
    # @dataclass
    # class Dataconfig:
    #     dataset_path: str = "/home/tien/playground_facereconstruction/data/react_2024"
    #     batch_size: int = 32
    #     num_workers: int = 8
    #     img_size: int = 256
    #     crop_size: int = 224
    #     clip_length: int = 256
    #     test_extend_factor: int = 5

    dataset = ReactionDataset(
        "/home/tien/playground_facereconstruction/data/react_2024",
        "train",
        img_size=256,
        crop_size=224,
        clip_length=256,
        load_audio=False,
        load_video_s=False,
        load_video_l=False,
        load_emotion_s=True,
        load_emotion_l=True,
        load_3dmm_s=True,
        load_3dmm_l=True,
        load_ref=False,
        repeat_mirrored=False,
    )
    import time

    t0 = time.time()
    for i, batch in enumerate(dataset):
        print(i, time.time() - t0, "seconds")
        t0 = time.time()
