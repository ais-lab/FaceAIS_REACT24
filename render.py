import os
import time

import cv2
import numpy as np
import torch
from skimage.io import imsave
from torchvision import transforms

from external.FaceVerse import get_faceverse
from external.FaceVerse.OpenSeeFace.tracker import Tracker
from external.PIRender import FaceGenerator
import external.FaceVerse.losses as losses
from external.FaceVerse.util_function import get_length, ply_from_array_color


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def _fix_image(image):
    if image.max() < 30.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image


def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)
    return semantic.transpose(1, 2)


class IncrementalFrame:
    def __init__(self) -> None:
        self.frames = []
        self.frame_names = []

    def add(self, frame, name):
        self.frames.append(frame)
        self.frame_names.append(name)

    def reset(self):
        self.frames = []
        self.frame_names = []

    def length(self):
        return len(self.frames)


class Render(object):
    """Computes and stores the average and current value"""

    def __init__(self, device="cpu"):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = (
            torch.from_numpy(np.load("external/FaceVerse/reference_full.npy"))
            .float()
            .view(1, -1)[:, :150]
        )
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()
        checkpoint = torch.load("external/PIRender/cur_model_fold.pth")
        self.pi_render.load_state_dict(checkpoint["state_dict"])

        self.mean_face = (
            torch.FloatTensor(
                np.load("external/FaceVerse/mean_face.npy").astype(np.float32)
            )
            .view(1, 1, -1)
            .to(device)
        )
        self.std_face = (
            torch.FloatTensor(
                np.load("external/FaceVerse/std_face.npy").astype(np.float32)
            )
            .view(1, 1, -1)
            .to(device)
        )

        self._reverse_transform_3dmm = transforms.Lambda(lambda e: e + self.mean_face)

        self.fake_video = IncrementalFrame()

    def rendering(
        self, path, ind, listener_vectors, speaker_video_clip, listener_reference
    ):
        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = (
            listener_vectors[:, :52].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.rot_tensor = (
            listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.trans_tensor = (
            listener_vectors[:, 55:].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.id_tensor = (
            self.id_tensor.view(1, 150)
            .repeat(T, 1)
            .view(T, 150)
            .to(listener_vectors.get_device())
        )

        pred_dict = self.faceverse(
            self.faceverse.get_packed_tensors(), render=True, texture=False
        )
        rendered_img_r = pred_dict["rendered_img"]
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(
            listener_vectors.get_device()
        )
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(
            listener_vectors.shape[0], 1, 1
        ).view(listener_vectors.shape[0], C, H, W)

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[
                    i * duration : (i + 1) * duration
                ]
                semantics_copy = semantics[i * duration : (i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration :]
                semantics_copy = semantics[i * duration :]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict["fake_image"]
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(
            os.path.join(path, ind + "_val.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            (672, 224),
        )
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

    def single_frame_render_mesh(
        self,
        path: str,
        name: str,
        facial_3dmm_vector: torch.Tensor,
        # speaker_img: torch.Tensor,
        # listener_img: torch.Tensor,
        # listener_reference: torch.Tensor,
        is_save=True,
    ):
        # 3D video
        T = facial_3dmm_vector.shape[0]
        facial_3dmm_vector = self._reverse_transform_3dmm(facial_3dmm_vector)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = (
            facial_3dmm_vector[:, :52].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.rot_tensor = (
            facial_3dmm_vector[:, 52:55].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.trans_tensor = (
            facial_3dmm_vector[:, 55:].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.id_tensor = (
            self.id_tensor.view(1, 150)
            .repeat(T, 1)
            .view(T, 150)
            .to(facial_3dmm_vector.get_device())
        )

        pred_dict = self.faceverse(
            self.faceverse.get_packed_tensors(), render=True, texture=False
        )
        rendered_img_r = pred_dict["rendered_img"]
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        if is_save:
            # write the first frame of the rendered_img_r to a image with the name of name
            os.makedirs(path, exist_ok=True)
            imsave(os.path.join(path, name + ".png"), rendered_img_r[0])

        return rendered_img_r[0]

    def single_frame_render_fake(
        self,
        path: str,
        name: str,
        facial_3dmm_vector: torch.Tensor,
        reference_img: torch.Tensor,
        is_final: bool = False,
    ):
        # add the frame to the fake_video
        self.fake_video.add(facial_3dmm_vector, name)

        # duration = listener_vectors.shape[0] // 20
        duration = 5  # 750/20 = 37.5
        if is_final:
            duration = self.fake_video.length()

        # if incremental frame len is over a threshold, then render save frame and reset
        if len(self.fake_video.frames) >= duration:
            listener_vectors = torch.stack(self.fake_video.frames, dim=0)
            frame_names = self.fake_video.frame_names

            listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]
            semantics = transform_semantic(listener_vectors.detach()).to(
                listener_vectors.get_device()
            )
            C, H, W = reference_img.shape
            output_dict_list = []

            listener_reference_frames = reference_img.repeat(
                listener_vectors.shape[0], 1, 1
            ).view(listener_vectors.shape[0], C, H, W)

            listener_reference_copy = listener_reference_frames
            semantics_copy = semantics

            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict["fake_image"]
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)
            listener_videos = np.concatenate(output_dict_list, axis=0)

            # save each frame to the path and name
            os.makedirs(path, exist_ok=True)
            for i in range(listener_videos.shape[0]):
                frame = listener_videos[i]
                frame = frame[:, :, ::-1]  # rotate the color channel
                imsave(os.path.join(path, frame_names[i] + ".png"), frame)

            self.fake_video.reset()

    def rendering_for_fid(
        self,
        path,
        ind,
        listener_vectors,  # for generate video and fake fid
        speaker_video_clip,  # for generate video
        listener_reference,  # for generate video
        listener_video_clip,  # for real fid
    ):
        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = (
            listener_vectors[:, :52].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.rot_tensor = (
            listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.trans_tensor = (
            listener_vectors[:, 55:].view(T, -1).to(listener_vectors.get_device())
        )
        self.faceverse.id_tensor = (
            self.id_tensor.view(1, 150)
            .repeat(T, 1)
            .view(T, 150)
            .to(listener_vectors.get_device())
        )

        pred_dict = self.faceverse(
            self.faceverse.get_packed_tensors(), render=True, texture=False
        )
        rendered_img_r = pred_dict["rendered_img"]
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(
            listener_vectors.get_device()
        )
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(
            listener_vectors.shape[0], 1, 1
        ).view(listener_vectors.shape[0], C, H, W)

        for i in range(20):  # why 20?
            if i != 19:
                listener_reference_copy = listener_reference_frames[
                    i * duration : (i + 1) * duration
                ]
                semantics_copy = semantics[i * duration : (i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration :]
                semantics_copy = semantics[i * duration :]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict["fake_image"]
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        if not os.path.exists(os.path.join(path, "results_videos")):
            os.makedirs(os.path.join(path, "results_videos"))
        out = cv2.VideoWriter(
            os.path.join(path, "results_videos", ind + "_val.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            (672, 224),
        )
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

        listener_video_clip = torch_img_to_np2(listener_video_clip)

        path_real = os.path.join(path, "fid", "real")
        if not os.path.exists(path_real):
            os.makedirs(path_real)
        path_fake = os.path.join(path, "fid", "fake")
        if not os.path.exists(path_fake):
            os.makedirs(path_fake)

        for i in range(0, rendered_img_r.shape[0], 30):  # default 30, let's change to 1
            cv2.imwrite(
                os.path.join(path_fake, ind + "_" + str(i + 1) + ".png"),
                listener_videos[i],
            )
            cv2.imwrite(
                os.path.join(path_real, ind + "_" + str(i + 1) + ".png"),
                listener_video_clip[i],
            )


class Extractor3DMM:
    def __init__(self, device="cpu"):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = (
            torch.from_numpy(np.load("external/FaceVerse/reference_full.npy"))
            .float()
            .view(1, -1)[:, :150]
        )
        
        self.mean_face = (
            torch.FloatTensor(
                np.load("external/FaceVerse/mean_face.npy").astype(np.float32)
            )
            .view(1, 1, -1)
            .to(device)
        )

        self._reverse_transform_3dmm = transforms.Lambda(lambda e: e + self.mean_face)

        self.lm_weights = losses.get_lm_weights(device)

        self.tracker = Tracker(
            640,
            480,
            threshold=None,
            max_threads=1,
            max_faces=1,
            discard_after=10,
            scan_every=30,
            silent=True,
            model_type=4,
            model_dir="external/FaceVerse/OpenSeeFace/models",
            no_gaze=True,
            detection_threshold=0.6,
            use_retinaface=1,
            max_feature_updates=900,
            static_model=False,
            try_hard=0,
        )

        self.device = device

    def init_optim_with_id(self, learning_rate_landmark=1e-2, learning_rate_diff=1e-2):
        rigid_optimizer = torch.optim.Adam(
            [
                self.faceverse.get_rot_tensor(),
                self.faceverse.get_trans_tensor(),
                self.faceverse.get_id_tensor(),
                self.faceverse.get_exp_tensor(),
            ],
            lr=learning_rate_landmark,
        )
        nonrigid_optimizer = torch.optim.Adam(
            [
                self.faceverse.get_id_tensor(),
                self.faceverse.get_exp_tensor(),
                self.faceverse.get_gamma_tensor(),
                self.faceverse.get_tex_tensor(),
                self.faceverse.get_rot_tensor(),
                self.faceverse.get_trans_tensor(),
            ],
            lr=learning_rate_diff,
        )  # TODO: change this
        return rigid_optimizer, nonrigid_optimizer

    def extract(self, frame, frame_id):

        tar_size = 224  # size for rendering window. We use a square window.
        id_reg_w = 1e-3  # weight for id coefficient regularizer
        exp_reg_w = 1.5e-4  # weight for expression coefficient regularizer
        tex_reg_w = 3e-4  # help='weight for texture coefficient regularizer'
        tex_w = 1  # weight for texture reflectance loss.
        rgb_loss_w = 1.6  # weight for rgb loss
        lm_loss_w = 3e3  # weight for landmark loss
        self.lms = np.zeros((66, 2), dtype=np.int64)
        
        # 0.01 s per frame
        face_track = self.tracker.predict(frame)
        
        
        if len(face_track) == 0:
            return None

        self.lms = (face_track[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
        self.lms = self.lms[:, [1, 0]]

        
        if frame_id == 0:
            self.border = 500
            self.half_length = int(get_length(self.lms))
            self.crop_center = self.lms[29].copy() + self.border
            print("First frame:", self.half_length, self.crop_center)
            self.rigid_optimizer, self.nonrigid_optimizer = self.init_optim_with_id()

            num_iters_rf = 500
            num_iters_rnf = 500

        else:
            num_iters_rf = 5
            num_iters_rnf = 5
        
        # preprocess time cost 0.001s
        frame_b = cv2.copyMakeBorder(
            frame, self.border, self.border, self.border, self.border, cv2.BORDER_CONSTANT, value=0
        )
        align = cv2.resize(
            frame_b[
                self.crop_center[1] - self.half_length : self.crop_center[1] + self.half_length,
                self.crop_center[0] - self.half_length : self.crop_center[0] + self.half_length,
            ],
            (tar_size, tar_size),
            cv2.INTER_AREA,
        )
        resized_lms = (
            (self.lms - (self.crop_center - self.half_length - self.border)[np.newaxis, :])
            / self.half_length
            / 2
            * tar_size
        )
        resized_lms = resized_lms.astype(np.int64)

        self.lms = (
            torch.from_numpy(resized_lms[np.newaxis, :, :])
            .type(torch.float32)
            .to(self.device)
        )
        img_tensor = (
            torch.from_numpy(align[np.newaxis, ...]).type(torch.float32).to(self.device)
        )

        start = time.time()
        for i in range(num_iters_rf):
            self.rigid_optimizer.zero_grad()

            pred_dict = self.faceverse(
                self.faceverse.get_packed_tensors(), render=False, texture=False
            )

            lm_loss_val = losses.lm_loss(
                pred_dict["lms_proj"], self.lms, self.lm_weights, img_size=tar_size
            )
            exp_reg_loss = losses.get_l2(self.faceverse.get_exp_tensor())
            id_reg_loss = losses.get_l2(self.faceverse.get_id_tensor())
            total_loss = (
                lm_loss_w * lm_loss_val
                + id_reg_loss * id_reg_w
                + exp_reg_loss * exp_reg_w
            )

            total_loss.backward()
            self.rigid_optimizer.step()

            with torch.no_grad():
                self.faceverse.exp_tensor[self.faceverse.exp_tensor < 0] *= 0
        # print("Rigid fitting time:", time.time() - start)
        
        start = time.time()
        for i in range(num_iters_rnf):
            self.nonrigid_optimizer.zero_grad()

            pred_dict = self.faceverse(
                self.faceverse.get_packed_tensors(), render=True, texture=True
            )
            rendered_img = pred_dict["rendered_img"]
            lms_proj = pred_dict["lms_proj"]
            face_texture = pred_dict["face_texture"]
            mask = rendered_img[:, :, :, 3].detach()

            lm_loss_val = losses.lm_loss(
                lms_proj, self.lms, self.lm_weights, img_size=tar_size
            )
            photo_loss_val = losses.photo_loss(
                rendered_img[:, :, :, :3], img_tensor, mask > 0
            )
            exp_reg_loss = losses.get_l2(self.faceverse.get_exp_tensor())
            id_reg_loss = losses.get_l2(self.faceverse.get_id_tensor())
            tex_reg_loss = losses.get_l2(self.faceverse.get_tex_tensor())
            tex_loss_val = losses.reflectance_loss(
                face_texture, self.faceverse.get_skinmask()
            )

            loss = (
                lm_loss_val * lm_loss_w
                + id_reg_loss * id_reg_w
                + exp_reg_loss * exp_reg_w
                + tex_reg_loss * tex_reg_w
                + tex_loss_val * tex_w
                + photo_loss_val * rgb_loss_w
            )

            loss.backward()
            self.nonrigid_optimizer.step()

            with torch.no_grad():
                self.faceverse.exp_tensor[self.faceverse.exp_tensor < 0] *= 0
        # print("Nonrigid fitting time:", time.time() - start)
        
        return self.create_3dmm_vector()
    
    def create_3dmm_vector(self):
        return torch.cat(
            [
                self.faceverse.get_exp_tensor(),
                self.faceverse.get_rot_tensor(),
                self.faceverse.get_trans_tensor(),
            ],
            dim=1,
        )

    def render(self, facial_3dmm_vector):
        # 3D video
        T = facial_3dmm_vector.shape[0]
        facial_3dmm_vector = self._reverse_transform_3dmm(facial_3dmm_vector)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = (
            facial_3dmm_vector[:, :52].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.rot_tensor = (
            facial_3dmm_vector[:, 52:55].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.trans_tensor = (
            facial_3dmm_vector[:, 55:].view(T, -1).to(facial_3dmm_vector.get_device())
        )
        self.faceverse.id_tensor = (
            self.id_tensor.view(1, 150)
            .repeat(T, 1)
            .view(T, 150)
            .to(facial_3dmm_vector.get_device())
        )

        pred_dict = self.faceverse(
            self.faceverse.get_packed_tensors(), render=True, texture=False
        )
        rendered_img_r = pred_dict["rendered_img"]
        rendered_img_r = np.clip(rendered_img_r.cpu().detach().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        return rendered_img_r[0]
