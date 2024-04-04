import argparse
import json
import os
import re

from PIL import Image


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def render_final_video(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith(".json"):
            render_name = file.split(".")[0]

            frame_dir = os.path.join(output_dir, render_name)
            os.makedirs(frame_dir, exist_ok=True)

            with open(os.path.join(output_dir, file), "r") as f:
                data = json.load(f)

            speaker_gt_frame = data["sp_gt_frame_address"]
            listener_gt_frame = data["lt_gt_frame_address"]
            mesh_frame_dir = data["mesh_video_address"]
            fake_frame_dir = data["fake_video_address"]
            if not os.path.isdir(mesh_frame_dir):
                mesh_frame = [
                    img
                    for img in os.listdir(os.path.join(output_dir, mesh_frame_dir))
                    if img.endswith(".png")
                ]
                fake_frame = [
                    img
                    for img in os.listdir(os.path.join(output_dir, fake_frame_dir))
                    if img.endswith(".png")
                ]
            
            else:   
                mesh_frame = [
                    img
                    for img in os.listdir( mesh_frame_dir)
                    if img.endswith(".png")
                ]
                fake_frame = [
                    img
                    for img in os.listdir(fake_frame_dir)
                    if img.endswith(".png")
                ]

            mesh_frame.sort(key=natural_keys)
            fake_frame.sort(key=natural_keys)

            for frame_idx in range(len(mesh_frame)):
                if not os.path.isdir(mesh_frame_dir):
                    mesh_img = os.path.join(
                        output_dir, mesh_frame_dir, mesh_frame[frame_idx]
                    )
                    fake_img = os.path.join(
                        output_dir, fake_frame_dir, fake_frame[frame_idx]
                    )
                else:
                    mesh_img = os.path.join(
                        mesh_frame_dir, mesh_frame[frame_idx]
                    )
                    
                    fake_img = os.path.join(
                        fake_frame_dir, fake_frame[frame_idx]
                    )
                speaker_gt_img = speaker_gt_frame[frame_idx]
                listener_gt_img = listener_gt_frame[frame_idx]

                # read image use pil
                mesh_img = Image.open(mesh_img)
                fake_img = Image.open(fake_img)
                speaker_gt_img = Image.open(speaker_gt_img)
                listener_gt_img = Image.open(listener_gt_img)

                # combine image
                combined_img = Image.new(
                    "RGB", (mesh_img.width * 4, mesh_img.height * 1)
                )

                combined_img.paste(speaker_gt_img, (0, 0))
                combined_img.paste(listener_gt_img, (mesh_img.width, 0))
                combined_img.paste(mesh_img, (mesh_img.width * 2, 0))
                combined_img.paste(fake_img, (mesh_img.width * 3, 0))

                # save image
                combined_img.save(os.path.join(frame_dir, f"frame_{frame_idx}.png"))

            # create video with high bit rate
            os.system(
                f"ffmpeg -r 30 -i {frame_dir}/frame_%d.png -b:v 10000k -vcodec mpeg4 -y {frame_dir}.mp4"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=str, default="output/")

    args = parser.parse_args()
    print("RENDERING FINAL VIDEO...")
    render_final_video(args.output_dir)
