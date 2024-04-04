import os

import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchvision.io import ImageReadMode, read_image


def compute_fid(
    output_dir=None,
    device="cuda",
    skip=30,
    real_path: list = None,
    fake_path: list = None,
):
    if output_dir is not None:
        real_dir = output_dir + "/real"
        fake_dir = output_dir + "/fake"

        assert os.path.exists(real_dir), "real_dir does not exist"
        assert os.path.exists(fake_dir), "fake_dir does not exist"
        assert len(os.listdir(real_dir)) == len(
            os.listdir(fake_dir)
        ), "real and fake dir have different number of files"

        pair = [
            (os.path.join(real_dir, file), os.path.join(fake_dir, file))
            for file in os.listdir(real_dir)
        ]

    if real_path is not None and fake_path is not None:
        pair = list(zip(real_path, fake_path))

    # breakpoint()

    reals = []
    fakes = []

    for idx, (real, fake) in enumerate(pair):
        if idx % skip != 0:
            continue

        real = read_image(real, ImageReadMode.RGB)
        fake = read_image(fake, ImageReadMode.RGB)

        reals.append(real)
        fakes.append(fake)

    real_tensor = torch.stack(reals)
    fake_tensor = torch.stack(fakes)

    real_tensor = real_tensor.to(device)
    fake_tensor = fake_tensor.to(device)

    fid = FID(feature=2048, normalize=False)
    fid = fid.to(device)

    fid.update(fake_tensor, real=False)
    fid.update(real_tensor, real=True)

    val = fid.compute()
    return val


if __name__ == "__main__":
    print(
        "FID: ",
        compute_fid(
            output_dir="/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/vq_emotion/fid/",
            device="cuda",
            skip=1,
        ),
    )
