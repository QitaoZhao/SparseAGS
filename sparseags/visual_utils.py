import os
import tqdm
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from kiui.mesh import Mesh
from kiui.cam import OrbitCamera
from kiui.op import safe_normalize
from kiui.lpips import LPIPS

from sparseags.mesh_utils.mesh_renderer import Renderer
from sparseags.cam_utils import OrbitCamera
from sparseags.render_utils.gs_renderer import CustomCamera


class GUI:
	def __init__(self, opt):
		self.opt = opt
		self.W = opt.W
		self.H = opt.H
		self.wogui = opt.wogui # disable gui and run in cmd
		self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
		self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg
		# self.bg_color = torch.zeros(3, dtype=torch.float32).cuda() # black bg

		self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
		self.need_update = True # camera moved, should reset accumulation
		self.light_dir = np.array([0, 0])
		self.ambient_ratio = 0.5

		# auto-rotate
		self.auto_rotate_cam = False
		self.auto_rotate_light = False
		
		self.mode = opt.mode
		self.render_modes = ['albedo', 'depth', 'normal', 'lambertian']

		# load mesh
		self.mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir)

		if not opt.force_cuda_rast and (self.wogui or os.name == 'nt'):
			self.glctx = dr.RasterizeGLContext()
		else:
			self.glctx = dr.RasterizeCudaContext()
	
	def step(self):

		if not self.need_update:
			return
	
		starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		starter.record()

		# do MVP for vertices
		pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
		proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
		
		v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
		v_clip = v_cam @ proj.T

		rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.H, self.W))

		alpha = (rast[..., 3:] > 0).float()
		alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
		
		if self.mode == 'depth':
			depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
			depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
			buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
		else:
			# use vertex color if exists
			if self.mesh.vc is not None:
				albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
			# use texture image
			else:
				texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft)
				albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]

			albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
			albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).clamp(0, 1) # [1, H, W, 3]
			if self.mode == 'albedo':
				albedo = albedo * alpha + self.bg_color * (1 - alpha)
				buffer = albedo[0].detach().cpu().numpy()
			else:
				normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
				normal = safe_normalize(normal)
				if self.mode == 'normal':
					normal_image = (normal[0] + 1) / 2
					normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
					buffer = normal_image.detach().cpu().numpy()
				elif self.mode == 'lambertian':
					light_d = np.deg2rad(self.light_dir)
					light_d = np.array([
						np.cos(light_d[0]) * np.sin(light_d[1]),
						-np.sin(light_d[0]),
						np.cos(light_d[0]) * np.cos(light_d[1]),
					], dtype=np.float32)
					light_d = torch.from_numpy(light_d).to(albedo.device)
					lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
					albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
					buffer = albedo[0].detach().cpu().numpy()

		ender.record()
		torch.cuda.synchronize()
		t = starter.elapsed_time(ender)

		self.render_buffer = buffer
		self.need_update = False

		if self.auto_rotate_cam:
			self.cam.orbit(5, 0)
			self.need_update = True
		
		if self.auto_rotate_light:
			self.light_dir[1] += 3
			self.need_update = True


def vis_output(camera_data, mesh_path=None, save_path=None, num_views=8):
	parser = argparse.ArgumentParser()
	parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
	parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
	parser.add_argument('--W', type=int, default=256, help="GUI width")
	parser.add_argument('--H', type=int, default=256, help="GUI height")
	parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
	parser.add_argument('--fovy', type=float, default=49.1, help="default GUI camera fovy")
	parser.add_argument("--wogui", type=bool, default=True, help="disable all dpg GUI")
	parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
	parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
	parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")
	parser.add_argument('--idx', type=int, default=0, help="GUI height")
	parser.add_argument('--config', default='configs/navi.yaml', type=str, help='Path to config directory, which contains image.yaml')
	args, extras = parser.parse_known_args()

	# override default config from cli
	opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
	data = camera_data

	cameras = [CustomCamera(cam_params) for cam_params in data.values()]
	cams = [(cam.c2w, cam.perspective, cam.focal_length) for cam in cameras]
	img_paths = [v["filepath"] for k, v in data.items()]

	opt.mesh = mesh_path
	opt.trainable_texture = False
	renderer = Renderer(opt).to(torch.device("cuda"))

	lpips_loss = LPIPS(net='vgg').cuda()
	mse_losses = []
	lpips_losses = []
	flags = [int(v["flag"]) for k, v in data.items()]
	images = np.zeros((2, num_views, 256, 256, 3), dtype=np.uint8)

	for i in tqdm.tqdm(range(len(cams))):

		img_path = img_paths[i]

		img = Image.open(img_path)
		if img.mode == 'RGBA':
			img = np.asarray(img, dtype=np.uint8).copy()
			img[img[:, :, -1] <= 255*0.9] = [255., 255., 255., 255.] # thresholding background
			img = img[:, :, :3]

		gt_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

		images[0, i] = img

		with torch.no_grad():
			out = renderer.render(*cams[i][:2], 256, 256, ssaa=1)

		# rgb loss
		image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8)
		pred_tensor = out["image"].permute(2, 0, 1).float().unsqueeze(0).cuda()
		# obj_scale = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach().sum().float()
		obj_scale = (out["alpha"] > 0).detach().sum().float()
		obj_scale /= 256 ** 2
		
		images[1, i] = image
		with torch.no_grad():
			mse_losses.append(F.mse_loss(pred_tensor, gt_tensor).squeeze().cpu().numpy() / obj_scale.item())
			lpips_losses.append(lpips_loss(pred_tensor, gt_tensor).squeeze().cpu().numpy() / obj_scale.item())

	mean_mse = np.mean(np.array(mse_losses)[:num_views])
	mean_lpips = np.mean(np.array(lpips_losses)[:num_views])

	num_frames = 2 * num_views
	cmap = plt.get_cmap("hsv")
	num_rows = 2
	num_cols = num_views
	plt.subplots_adjust(top=0.2)
	figsize = (num_cols * 2, num_rows * 2.2)
	fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
	fig.suptitle(f"Avg MSE: {mean_mse:.4f}, Avg LPIPS: {mean_lpips:.4f}", fontsize=16, y=0.97)
	axs = axs.flatten()
	for i in range(num_rows * num_cols):
		if i < num_frames:
			axs[i].imshow(images.reshape(-1, 256, 256, 3)[i])
			for s in ["bottom", "top", "left", "right"]:
				if i % num_views <= num_views - 1:
					if not flags[i%num_views]:
						axs[i].spines[s].set_color("red")
					else:
						axs[i].spines[s].set_color("green")
				else:
					axs[i].spines[s].set_color(cmap(i / (num_frames)))
				axs[i].spines[s].set_linewidth(5)
			axs[i].set_xticks([])
			axs[i].set_yticks([])

			if i >= num_views:
				axs[i].set_xlabel(f'MSE: {mse_losses[i%num_views]:.4f}\nLPIPS: {lpips_losses[i%num_views]:.4f}', fontsize=10)
		else:
			axs[i].axis("off")
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)
	print(f"Visualization file written to {save_path}")

	out_dir = save_path.replace('vis.png', 'reprojections') 
	os.makedirs(out_dir, exist_ok=True)

	for i in range(num_views):
		gt = Image.fromarray(images[0, i])
		pred = Image.fromarray(images[1, i])
		gt.save(os.path.join(out_dir, f"gt_{i}.png"))
		pred.save(os.path.join(out_dir, f"pred_{i}.png"))

	return np.array(lpips_losses), np.array(mse_losses)

	