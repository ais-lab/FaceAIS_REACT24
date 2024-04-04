import argparse
import json
import os
import re

import torch

from metrics.compute import compute


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_test", type=str, default="val")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    val_test = args.val_test

    listener_pred = torch.load(os.path.join(output_dir, "listener_pred_emotion.pt")).cpu()
    listener_gt = torch.load(os.path.join(output_dir, "listener_gt.pt")).cpu()
    speaker_gt = torch.load(os.path.join(output_dir, "speaker_gt.pt")).cpu()

    listener_pred = listener_pred.reshape(
        (listener_gt.shape[0], -1, listener_pred.shape[-2], listener_pred.shape[-1])
    )

    real_path_list = []
    fake_path_list = []

    for file in os.listdir(output_dir):
        if file.endswith(".json"):
            render_name = file.split(".")[0]

            frame_dir = os.path.join(output_dir, render_name)
            os.makedirs(frame_dir, exist_ok=True)

            with open(os.path.join(output_dir, file), "r") as f:
                data = json.load(f)

            listener_gt_frame = data["lt_gt_frame_address"]
            fake_frame_dir = data["fake_video_address"]

            if not os.path.isdir(fake_frame_dir):
                fake_frame = [
                    img
                    for img in os.listdir(os.path.join(output_dir, fake_frame_dir))
                    if img.endswith(".png")
                ]
            else:
                fake_frame = [
                    img
                    for img in os.listdir(fake_frame_dir)
                    if img.endswith(".png")
                ]

            fake_frame.sort(key=natural_keys)

            if not os.path.isdir(fake_frame_dir):
                _fake_frame = [
                    os.path.join(fake_frame_dir, img) for img in fake_frame
                ]
            else:
                _fake_frame = [
                    os.path.join(output_dir, fake_frame_dir, img) for img in fake_frame
                ]

            real_path_list.extend(listener_gt_frame)
            fake_path_list.extend(_fake_frame)

    print(f"Computing metrics... for {val_test} set!")
    print("It might take a while, please be patient.")
    print(f"Num frame for FID: {len(real_path_list)}")

    # breakpoint()

    max_frame = 5000
    if len(real_path_list) > max_frame:
        real_path_list = real_path_list[:max_frame]
        fake_path_list = fake_path_list[:max_frame]

    metrics = compute(
        dataset_path=dataset_path,
        listener_pred=listener_pred,
        speaker_gt=speaker_gt,
        listener_gt=listener_gt,
        val_test=val_test,
        device="cuda",
        p=12,
        list_real=real_path_list,
        list_fake=fake_path_list,
    )

    # print("TLCC: ", metrics.TLCC.avg)
    # print("FRC: ", metrics.FRC.avg)
    # print("FRD: ", metrics.FRD.avg)
    # print("FRDvs: ", metrics.FRDvs.avg)
    # print("FRVar: ", metrics.FRVar.avg)
    # print("smse: ", metrics.smse.avg)
    # print("FRRea: ", metrics.FRRea.avg)

    # write to file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"TLCC: {metrics.TLCC.avg}\n")
        f.write(f"FRC: {metrics.FRC.avg}\n")
        f.write(f"FRD: {metrics.FRD.avg}\n")
        f.write(f"FRDvs: {metrics.FRDvs.avg}\n")
        f.write(f"FRVar: {metrics.FRVar.avg}\n")
        f.write(f"smse: {metrics.smse.avg}\n")
        f.write(f"FRRea: {metrics.FRRea.avg}\n")
