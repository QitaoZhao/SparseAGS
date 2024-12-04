import os
import cv2
import sys
import json
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg
from liegroups.torch import SE3

import sys 
sys.path.append('./')

from sparseags.cam_utils import orbit_camera, OrbitCamera, mat2latlon, find_mask_center_and_translate
from sparseags.render_utils.gs_renderer import Renderer, Camera, FoVCamera, CustomCamera
from sparseags.mesh_utils.grid_put import mipmap_linear_grid_put_2d
from sparseags.mesh_utils.mesh import safe_normalize


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
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer.enable_dino = self.opt.lambda_dino > 0
        self.renderer.gaussians.enable_dino = self.opt.lambda_dino > 0
        self.renderer.gaussians.dino_feat_dim = 36
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data
        self.load_input(self.opt.camera_path, self.opt.order_path)

        self.cam = OrbitCamera(opt.W, opt.H, r=3, fovy=opt.fovy)

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts, radius=0.3, mode='sphere')  # 0.5 for radius 3

            # initialize gaussians to a carved voxel 
            # self.renderer.initialize(num_pts=self.opt.num_pts, radius=0.5, cameras=self.cams, masks=self.input_mask, mode='carve')  # 0.5

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
        self.renderer.gaussians.training_setup(self.opt)
        # do progressive sh-level
        self.renderer.gaussians.active_sh_degree = 0
        self.optimizer = self.renderer.gaussians.optimizer

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None
        self.enable_dino = self.opt.lambda_dino > 0 

        # lazy load guidance model
        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from sparseags.guidance_utils.zero123_6d_utils import Zero123
            self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
            self.guidance_zero123.opt = self.opt
            self.guidance_zero123.num_views = self.num_views

        # input image
        if self.input_img is not None:
            import torchvision.transforms as transforms
            from PIL import Image
            self.input_img_torch = torch.from_numpy(self.input_img).permute(0, 3, 1, 2).to(self.device)
            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(0, 3, 1, 2).to(self.device)

        # prepare embeddings
        with torch.no_grad():
            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            for choice in range(self.num_views):
                # For multiview training
                cur_cam = self.cams[choice]

                bg_size = self.renderer.gaussians.dino_feat_dim if self.enable_dino else 3
                bg_color = torch.ones(
                    bg_size,
                    dtype=torch.float32,
                    device="cuda",
                )
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                # rgb loss
                image = out["image"]
                loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img_torch[choice])

                # mask loss
                mask = out["alpha"]
                loss = loss + 1000 * step_ratio * F.mse_loss(mask, self.input_mask_torch[choice])

                # dino loss
                if self.enable_dino:
                    feature = out["feature"]
                    loss = loss + 1000 * step_ratio * F.mse_loss(feature, self.guidance_dino.embeddings[choice])

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            masks = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80)
            min_ver = max(-60 + np.array(self.opt.ref_polars).min(), -80)  # + - 30 for co3D
            max_ver = min(60 + np.array(self.opt.ref_polars).max(), 80)

            for _ in range(self.opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver) - self.opt.ref_polars[0]
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(
                    self.opt.ref_polars[0] + ver, 
                    self.opt.ref_azimuths[0] + hor, 
                    np.array(self.opt.ref_radii).mean() + radius,
                )

                # Azimuth
                # [-180, -135): -4, [-135, -90): -3, [-90, -45): -2, [-45, 0): -1
                # [0, 45): 0, [45, 90): 1, [90, 135): 2, [135, 180): 3. 
                # Elevation: [0, 90): 0 [-90, 0): 1
                idx_ver, idx_hor = int((self.opt.ref_polars[0]+ver) < 0), hor // 45

                flag = 0
                cx, cy = self.pp_pools[idx_ver, idx_hor+4].tolist()
                cnt = 0
                fx, fy = self.fx, self.fy

                # in each iter we modify cx, cy, fx, fy to make sure the rendered object is at the center and has a reasonable size
                while not flag:

                    if cnt >= 10:
                        # print(f"[ERROR] Something might be wrong here!")
                        break

                    flag_principal_point, flag_focal_length = 0, 0

                    # we modified the field of view. Otherwise, the rendered object will be too small
                    # cur_cam = FoVCamera(pose, render_resolution, render_resolution, self.fovy, self.fovx, self.cam.near, self.cam.far)
                    cur_cam = Camera(pose, render_resolution, render_resolution, fx, fy, cx, cy, self.cam.near, self.cam.far)

                    bg_size = self.renderer.gaussians.dino_feat_dim if self.enable_dino else 3
                    bg_color = torch.ones(bg_size, dtype=torch.float32, device="cuda") if np.random.rand() > self.opt.invert_bg_prob else torch.zeros(bg_size, dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color)

                    image = out["image"].unsqueeze(0) 
                    mask = out["alpha"].unsqueeze(0)
                    delta_xy = find_mask_center_and_translate(image.detach(), mask.detach()) / render_resolution * 256

                    # (1) check if the principal points are appropriate
                    if delta_xy[0].abs() > 10 or delta_xy[1].abs() > 10:
                        cx -= delta_xy[0]
                        cy -= delta_xy[1]
                        self.pp_pools[idx_ver, idx_hor+4] = torch.tensor([cx, cy])  # Update pp_pools
                    else:
                        flag_principal_point = 1

                    num_pixs_mask = (mask > 0.5).float().sum().item()
                    target_num_pixs = render_resolution ** 2 / (1.2 ** 2)

                    mask_to_compute = (mask > 0.5).squeeze().detach().cpu().numpy()
                    y_indices, x_indices = np.where(mask_to_compute > 0)
    
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        # return None or some indication that there's no object in the mask
                        continue
                    
                    # find the bounding box coordinates
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
    
                    bbox =  np.array([x1, y1, x2, y2])
                    extents = (bbox[2:] - bbox[:2]).max()
                    num_pixs_mask = extents ** 2

                    # (2) check if the focal lengths are appropriate
                    if abs(num_pixs_mask - target_num_pixs) > 0.05 * render_resolution ** 2:
                        if num_pixs_mask == 0:
                            pass
                        else:
                            fx = fx * np.sqrt(target_num_pixs / num_pixs_mask)
                            fy = fy * np.sqrt(target_num_pixs / num_pixs_mask)
                    else:
                        flag_focal_length = 1

                    if flag_principal_point * flag_focal_length == 1:
                        flag = 1

                    cnt += 1

                images.append(image)
                masks.append(mask)

            images = torch.cat(images, dim=0)

            if self.enable_zero123:
                target_RT = {
                    "c2w": pose,
                    "focal_length": np.array(fx, fy),
                }
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.batch_train_step(images, target_RT, self.cams, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            if self.enable_dino:
                loss_dino = self.guidance_dino.train_step(
                    images, 
                    out["feature"],
                    step_ratio=step_ratio if self.opt.anneal_timestep else None
                )
                loss = loss + self.opt.lambda_dino * loss_dino

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            latlons = [mat2latlon(cam.c2w[:3, 3]) for cam in self.cams]
            if self.opt.opt_cam:
                for i, cam in enumerate(self.cams):
                    w2c = cam.w2c @ SE3.exp(cam.cam_params.detach()).as_matrix()
                    w2c[:2, :3] *= -1
                    w2c[:2, 3] *= -1
                    self.camera_tracks[i].append(w2c.tolist())
            self.opt.ref_polars = [float(cam[0]) for cam in latlons]
            self.opt.ref_azimuths = [float(cam[1]) for cam in latlons]
            self.opt.ref_radii = [float(cam[2]) for cam in latlons]

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                # if self.step % self.opt.opacity_reset_interval == 0:
                #     self.renderer.gaussians.reset_opacity()

                if self.step % 100 == 0 and self.renderer.gaussians.max_sh_degree != 0:
                    self.renderer.gaussians.oneupSHdegree()

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
            for k, v in self.cam_params.items():
                self.cam_params[k]['opt_cam'] = 1
                self.cam_params[k]['flag'] = 1
        else:
            for k, v in self.cam_params.items():
                if int(self.cam_params[k]['flag']):
                    self.cam_params[k]['opt_cam'] = 1
                else:
                    self.cam_params[k]['opt_cam'] = 0

        img_paths = [v["filepath"] for k, v in self.cam_params.items() if v["flag"]]
        self.num_views = len(img_paths)
        print(f"[INFO] Number of views: {self.num_views}")

        for filepath in img_paths:
            print(filepath)

        images, masks = [], []

        for i in range(self.num_views):
            img = cv2.imread(img_paths[i], cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 3:
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(img, session=self.bg_remover)

            img = img.astype(np.float32) / 255.0

            # Non-integer cropping creates non-zero mask values
            input_mask = (img[..., 3:] > 0.5).astype(np.float32)

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

        self.cams = [CustomCamera(v, index=int(k), opt_pose=self.opt.opt_cam and v['opt_cam']) for k, v in self.cam_params.items() if v["flag"]]
        cam_centers = [mat2latlon(cam.camera_center) for cam in self.cams]
        self.opt.ref_polars = [float(cam[0]) for cam in cam_centers]
        self.opt.ref_azimuths = [float(cam[1]) for cam in cam_centers]
        self.opt.ref_radii = [float(cam[2]) for cam in cam_centers]
        self.fx = np.array([cam.fx for cam in self.cams], dtype=np.float32).mean()
        self.fy = np.array([cam.fy for cam in self.cams], dtype=np.float32).mean()
        self.cx = 128
        self.cy = 128
        if self.opt.opt_cam:
            self.camera_tracks = {}
            for i, cam in enumerate(self.cams):
                self.camera_tracks[i] = []

        # Azimuth Mapping: [-180, -135): -4, [-135, -90): -3, [-90, -45): -2, [-45, 0): -1,
        #                   [0, 45): 0, [45, 90): 1, [90, 135): 2, [135, 180): 3.
        # Elevation Mapping: [0, 90): 0, [-90, 0): 1.

        # Principal Point Pool: Tensor (2, 8, 2), where:
        #   - 2: Elevation groups, 8: Azimuth intervals, 2: x, y coordinates (init to 128).

        # we created a "pool" for principal points
        # we use these principal points to render image to make sure object is at the center
        self.pp_pools = torch.full((2, 8, 2), 128)
        if self.opt.opt_cam:
            self.renderer.gaussians.cam_params = [cam.cam_params for cam in self.cams[:] if cam.opt_pose]

    @torch.no_grad()
    def save_video(self, post_fix=None):
        xyz = self.renderer.gaussians._xyz
        center = self.renderer.gaussians._xyz.mean(dim=0)
        squared_distances = torch.sum((xyz - center) ** 2, dim=1)
        max_distance_squared = torch.max(squared_distances)
        radius = torch.sqrt(max_distance_squared) + 1.0
        radius = radius.detach().cpu().numpy()

        render_resolution = 256
        images = []
        frame_rate = 30
        image_size = (render_resolution, render_resolution)  # Size of each image
        video_path = self.opt.save_path + f'_rendered_video_{post_fix}.mp4'

        azimuth = np.arange(0, 360, 3, dtype=np.int32)

        for azi in tqdm.tqdm(azimuth):
            target = center.detach().cpu().numpy()
            pose = orbit_camera(-30, azi, radius, target=target)
            cur_cam = FoVCamera(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam)
            img = out["image"].detach().cpu().numpy() # [3, H, W] in [0, 1]
            img = np.transpose(img, (1, 2, 0))
            image = (img * 255).astype(np.uint8)
            images.append(image)

        images = np.stack(images, axis=0)
        # ~4 seconds, 120 frames at 30 fps
        import imageio
        imageio.mimwrite(video_path, images, fps=30, quality=8, macro_block_size=1)


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
            if self.enable_dino:
                feature = torch.zeros((h, w, self.renderer.gaussians.dino_feat_dim), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = FoVCamera(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                if self.enable_dino:
                    features = cur_out["feature"].unsqueeze(0) # [1, 384, 512, 512]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )

                if self.enable_dino:
                    features = features.view(features.shape[1], -1).permute(1, 0)[mask].contiguous()
                    cur_feature, _ = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    features,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

                if self.enable_dino:
                    feature[mask] += cur_feature[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            if self.enable_dino:
                feature[mask] = feature[mask] / cnt[mask].repeat(1, feature.shape[-1])

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            if self.enable_dino:
                feature = feature.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            # mesh.write(path)

            if self.enable_dino:
                feature[tuple(inpaint_coords.T)] = feature[tuple(search_coords[indices[:, 0]].T)]
                mesh.feature = torch.from_numpy(feature).to(self.device)

            mesh.write(path, self.enable_dino)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        if self.opt.opt_cam:
            for cam in self.cams:
                try:
                    self.cam_params[str(cam.index)]["R"] = cam.rotation.tolist()
                    self.cam_params[str(cam.index)]["T"] = cam.translation.tolist()
                except KeyError:
                    self.cam_params[f"{cam.index:03}"]["R"] = cam.rotation.tolist()
                    self.cam_params[f"{cam.index:03}"]["T"] = cam.translation.tolist()
        with open(self.opt.camera_path.replace(".json", "_updated.json"), "w") as file:
            json.dump(self.cam_params, file, indent=4)
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    gui.train(opt.iters)
