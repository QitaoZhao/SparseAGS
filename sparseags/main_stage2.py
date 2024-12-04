import os
import cv2
import json
import copy
import tqdm
import rembg
import torch
import torch.nn.functional as F
import numpy as np


import sys
sys.path.append('./')

from sparseags.cam_utils import orbit_camera, mat2latlon, find_mask_center_and_translate
from sparseags.render_utils.gs_renderer import CustomCamera
from sparseags.mesh_utils.mesh_renderer import Renderer


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H

        self.mode = "image"
        self.seed = 0

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_dino = None

        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_dino = False

        # renderer
        self.renderer = Renderer(opt).to(self.device)

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data
        self.load_input(self.opt.camera_path, self.opt.order_path)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        cameras = [CustomCamera(v, index=int(k)) for k, v in self.cam_params.items() if v["flag"]]
        cam_centers = [mat2latlon(cam.camera_center) for cam in cameras]
        self.opt.ref_polars = [float(cam[0]) for cam in cam_centers]
        self.opt.ref_azimuths = [float(cam[1]) for cam in cam_centers]
        self.opt.ref_radii = [float(cam[2]) for cam in cam_centers]
        self.cams = [(cam.c2w, cam.perspective, cam.focal_length) for cam in cameras]
        self.cam = copy.deepcopy(cameras[0])
        
        # Azimuth Mapping: [-180, -135): -4, [-135, -90): -3, [-90, -45): -2, [-45, 0): -1,
        #                   [0, 45): 0, [45, 90): 1, [90, 135): 2, [135, 180): 3.
        # Elevation Mapping: [0, 90): 0, [-90, 0): 1.

        # Principal Point Pool: Tensor (2, 8, 2), where:
        #   - 2: Elevation groups, 8: Azimuth intervals, 2: x, y coordinates (init to 128).

        # we created a "pool" for principal points
        # we use these principal points to render image to make sure object is at the center
        self.pp_pools = torch.full((2, 8, 2), 128)

        # The intrinsics is the average over all cams
        self.cam.fx = np.array([cam.fx for cam in cameras], dtype=np.float32).mean()
        self.cam.fy = np.array([cam.fy for cam in cameras], dtype=np.float32).mean()
        self.cam.cx = np.array([cam.cx for cam in cameras], dtype=np.float32).mean()
        self.cam.cy = np.array([cam.cy for cam in cameras], dtype=np.float32).mean()
        
        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None
        self.enable_dino = self.opt.lambda_dino > 0

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from sparseags.guidance_utils.zero123_6d_utils import Zero123
            self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        if self.guidance_dino is None and self.enable_dino:
            print(f"[INFO] loading dino...")
            from guidance.dino_utils import Dino
            self.guidance_dino = Dino(self.device, n_components=36, model_key="dinov2_vits14")
            self.guidance_dino.fit_pca(self.all_input_images)
            print(f"[INFO] loaded dino!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(0, 3, 1, 2).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(0, 3, 1, 2).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch.permute(0, 2, 3, 1).contiguous()

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

            if self.enable_dino:
                self.guidance_dino.embeddings = self.guidance_dino.get_dino_embeds(self.input_img_torch, upscale=True, reduced=True, learned_up=True)  # [8, 18, 18, 36]

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0

            ### known view
            for choice in range(self.num_views):
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(*self.cams[choice][:2], self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                # rgb loss
                image = out["image"] # [H, W, 3] in [0, 1]
                valid_mask = (out["alpha"] > 0).detach()
                loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last[choice] * valid_mask)

                if self.enable_dino:
                    feature = out["feature"]
                    loss = loss + F.mse_loss(feature * valid_mask, self.guidance_dino.embeddings[choice] * valid_mask)

            ### novel view (manual batch)
            render_resolution = 512
            images = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            # min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            # max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            # min_ver = max(min(-30, -30 + np.array(self.opt.ref_polars).min()), -80)
            # max_ver = min(max(30, 30 + np.array(self.opt.ref_polars).max()), 80)
            min_ver = max(-30 + np.array(self.opt.ref_polars).min(), -80)
            max_ver = min(30 + np.array(self.opt.ref_polars).max(), 80)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver) - self.opt.ref_polars[0]
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.ref_polars[0] + ver, self.opt.ref_azimuths[0] + hor, np.array(self.opt.ref_radii).mean() + radius)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))

                # Azimuth
                # [-180, -135): -4, [-135, -90): -3, [-90, -45): -2, [-45, 0): -1
                # [0, 45): 0, [45, 90): 1, [90, 135): 2, [135, 180): 3. 
                # Elevation: [0, 90): 0 [-90, 0): 1
                idx_ver, idx_hor = int((self.opt.ref_polars[0]+ver) < 0), hor // 45

                flag = 0
                cx, cy = self.pp_pools[idx_ver, idx_hor+4].tolist() 
                cnt = 0

                while not flag:

                    self.cam.cx = cx
                    self.cam.cy = cy

                    if cnt >= 5:
                        print(f"[ERROR] Something must be wrong here!")
                        break

                    # We modified the field of view. Otherwise, the rendered object will be too small
                    out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                    image = out["image"]
                    image = image.permute(2, 0, 1).contiguous().unsqueeze(0)
                    mask = out["alpha"] > 0
                    mask = mask.permute(2, 0, 1).contiguous().unsqueeze(0)
                    delta_xy = find_mask_center_and_translate(image.detach(), mask.detach()) / render_resolution * 256

                    if delta_xy[0].abs() > 10 or delta_xy[1].abs() > 10:
                        cx -= delta_xy[0]
                        cy -= delta_xy[1]
                        self.pp_pools[idx_ver, idx_hor+4] = torch.tensor([cx, cy])  # Update pp_pools
                        cnt += 1
                    else:
                        flag = 1

                images.append(image)

            images = torch.cat(images, dim=0)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8
            if self.enable_zero123:
                v1 = torch.stack([torch.tensor([radius]) + self.opt.ref_radii[0], torch.deg2rad(torch.tensor([ver]) + self.opt.ref_polars[0]), torch.deg2rad(torch.tensor([hor]) + self.opt.ref_azimuths[0])], dim=-1)   # polar,azimuth,radius are all actually delta wrt default
                v2 = torch.stack([torch.tensor(self.opt.ref_radii), torch.deg2rad(torch.tensor(self.opt.ref_polars)), torch.deg2rad(torch.tensor(self.opt.ref_azimuths))], dim=-1)
                angles = torch.rad2deg(self.guidance_zero123.angle_between(v1, v2)).to(self.device)
                choice = torch.argmin(angles.squeeze()).item()

                cond_RT = {
                    "c2w": self.cams[choice][0],
                    "focal_length": self.cams[choice][-1],
                }
                target_RT = {
                    "c2w": pose,
                    "focal_length": np.array(self.cam.fx, self.cam.fy),
                }
                cam_embed = self.guidance_zero123.get_cam_embeddings_6D(target_RT, cond_RT)

                # Additionally add an idx parameter to choose the correct viewpoints
                refined_images = self.guidance_zero123.refine(images, cam_embed, strength=strength, idx=choice).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)

            if self.enable_dino:
                loss_dino = self.guidance_dino.train_step(
                    images, 
                    out["feature"].permute(2, 0, 1).contiguous(),
                    step_ratio=step_ratio if self.opt.anneal_timestep else None
                )
                loss = loss + self.opt.lambda_dino * loss_dino

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True
    
    def load_input(self, camera_path, order_path=None):
        # load image
        print(f'[INFO] load data from {camera_path}...')

        if order_path is not None:
            with open(order_path, 'r') as f:
                indices = json.load(f)
        else:
            indices = None

        with open(camera_path, 'r') as f:
            data = json.load(f)

        self.cam_params = {}
        for k, v in data.items():
            if indices is None:
                self.cam_params[k] = data[k]
            else:
                if int(k) in indices or k in indices:
                    self.cam_params[k] = data[k]

            if self.opt.all_views:
                v['flag'] = 1

        img_paths = [v["filepath"] for k, v in self.cam_params.items() if v["flag"]]
        self.num_views = len(img_paths)
        print(f"[INFO] Number of views: {self.num_views}")

        for filepath in img_paths:
            print(filepath)

        images, masks = [], []

        for i in range(len(img_paths)):
            img = cv2.imread(img_paths[i], cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 3:
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(img, session=self.bg_remover)

            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0

            input_mask = img[..., 3:]
            # white bg
            input_img = img[..., :3] * input_mask + (1 - input_mask)
            # bgr to rgb
            input_img = input_img[..., ::-1].copy()

            images.append(input_img), masks.append(input_mask)

        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)
        self.input_img = images[:self.num_views]
        self.input_mask = masks[:self.num_views]
        self.all_input_images = images
    
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)

    gui.train(opt.iters_refine)
