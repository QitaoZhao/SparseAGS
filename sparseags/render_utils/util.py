import os
import gc
import copy
import tqdm
import argparse
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from kiui.lpips import LPIPS
from liegroups.torch import SE3

import sys 
sys.path.append('./')

from sparseags.render_utils.gs_renderer import CustomCamera
from sparseags.mesh_utils.mesh_renderer import Renderer
from sparseags.cam_utils import mat2latlon


def safe_normalize(x):
	return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)


def look_at(campos, target, opengl=True):
	if not opengl:
		forward_vector = safe_normalize(target - campos)
		up_vector = torch.tensor([0, 1, 0], dtype=campos.dtype, device=campos.device).expand_as(forward_vector)
		right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
		up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))
	else:
		forward_vector = safe_normalize(campos - target)
		up_vector = torch.tensor([0, 1, 0], dtype=campos.dtype, device=campos.device).expand_as(forward_vector)
		right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
		up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1))
	R = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
	return R


def orbit_camera(elevation, azimuth, radius=1.0, is_degree=True, target=None, opengl=True):
	"""Converts elevation & azimuth to a batch of camera pose matrices."""
	if is_degree:
		elevation = torch.deg2rad(elevation)
		azimuth = torch.deg2rad(azimuth)
	x = radius * torch.cos(elevation) * torch.sin(azimuth)
	y = -radius * torch.sin(elevation)
	z = radius * torch.cos(elevation) * torch.cos(azimuth)
	if target is None:
		target = torch.zeros(3, dtype=torch.float32, device=elevation.device)
	campos = torch.stack([x, y, z], dim=-1) + target
	R = look_at(campos, target.unsqueeze(0).expand_as(campos), opengl)
	T = torch.eye(4, dtype=torch.float32, device=elevation.device).unsqueeze(0).expand(campos.shape[0], -1, -1).clone()
	T[:, :3, :3] = R
	T[:, :3, 3] = campos
	return T


def render_and_compare(camera_data, mesh_path, save_path, num_views=8):
	parser = argparse.ArgumentParser()
	parser.add_argument('--object', type=str, help="path to mesh (obj, ply, glb, ...)")
	parser.add_argument('--path', type=str, help="path to mesh (obj, ply, glb, ...)")
	parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
	parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
	parser.add_argument('--W', type=int, default=256, help="GUI width")
	parser.add_argument('--H', type=int, default=256, help="GUI height")
	parser.add_argument("--wogui", type=bool, default=True, help="disable all dpg GUI")
	parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
	parser.add_argument("--config", default='configs/navi.yaml', help="path to the yaml config file")
	parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
	parser.add_argument('--fovy', type=float, default=49.1, help="default GUI camera fovy")
	args, extras = parser.parse_known_args()

	# override default config from cli
	opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
	data = camera_data

	opt.mesh = mesh_path
	opt.trainable_texture = False
	renderer = Renderer(opt).to(torch.device("cuda"))
	target = renderer.mesh.v.mean(dim=0)

	cameras = [CustomCamera(cam_params) for cam_params in data.values()]
	# cams = [(cam.c2w, cam.perspective, cam.focal_length) for cam in cameras]
	img_paths = [v["filepath"] for k, v in data.items()]
	flags = [int(v["flag"]) for k, v in data.items()]

	cam_centers = [mat2latlon(cam.camera_center - target) for idx, cam in enumerate(cameras) if flags[idx]]
	ref_polars = [float(cam[0]) for cam in cam_centers]
	ref_azimuths = [float(cam[1]) for cam in cam_centers]
	ref_radii = [float(cam[2]) for cam in cam_centers]

	base_cam = copy.copy(cameras[0])
	base_cam.fx = np.array([cam.fx for idx, cam in enumerate(cameras) if flags[idx]], dtype=np.float32).mean()
	base_cam.fy = np.array([cam.fy for idx, cam in enumerate(cameras) if flags[idx]], dtype=np.float32).mean()
	base_cam.cx = 128
	base_cam.cy = 128

	lpips_loss = LPIPS(net='vgg').cuda()
	elevation_range = (max([min(ref_polars) - 20, -89.9]), min([max(ref_polars) + 20, 89.9]))  
	azimuth_range = (-180, 180)  
	radius_range = (min(ref_radii) - 0.2, max(ref_radii) + 0.2)

	elevation_steps = torch.arange(elevation_range[0], elevation_range[1], 15, dtype=torch.float32)
	azimuth_steps = torch.arange(azimuth_range[0], azimuth_range[1], 15, dtype=torch.float32)
	radius_steps = torch.arange(radius_range[0], radius_range[1], 0.2, dtype=torch.float32)
	elevation_grid, azimuth_grid, radius_grid = torch.meshgrid(elevation_steps, azimuth_steps, radius_steps, indexing='ij')
	pose_grid = torch.stack((elevation_grid.flatten(), azimuth_grid.flatten(), radius_grid.flatten()), dim=1)

	poses = orbit_camera(pose_grid[:, 0], pose_grid[:, 1], pose_grid[:, 2], target=target.cpu())
	print("Number of hypotheses:", poses.shape[0])
	s1_steps = 128
	s2_steps = 256
	beta = 0.25
	chunk_size = 512

	for i in tqdm.tqdm(range(num_views)):
		if flags[i]:
			continue

		pose_grid = torch.stack((elevation_grid.flatten(), azimuth_grid.flatten(), radius_grid.flatten()), dim=1)

		poses = orbit_camera(pose_grid[:, 0], pose_grid[:, 1], pose_grid[:, 2], target=target.cpu())

		img_path = img_paths[i]
		base_cam.fx = cameras[i].fx
		base_cam.fy = cameras[i].fy
		perspectives = torch.from_numpy(base_cam.perspective).expand(pose_grid.shape[0], -1, -1)

		learnable_cam_params = torch.randn(pose_grid.shape[0], 6) * 1e-6
		learnable_cam_params.requires_grad_()

		loss_MSE_grid = np.zeros(pose_grid.shape[0])
		loss_LPIPS_grid = np.zeros(pose_grid.shape[0])
		loss = 0

		gt_img = Image.open(img_path)
		if gt_img.mode == 'RGBA':
			gt_img = np.asarray(gt_img, dtype=np.uint8).copy()
			gt_mask = (gt_img[..., 3:] > 128).astype(np.float32)
			gt_img[gt_img[:, :, -1] <= 255*0.9] = [255., 255., 255., 255.] # thresholding background
			gt_img = gt_img[:, :, :3]

		gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).cuda() / 255.
		gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).cuda()

		num_batches = pose_grid.shape[0] // chunk_size + int(pose_grid.shape[0]%chunk_size > 0)

		# Render images for visualization
		vis_img = torch.zeros(pose_grid.shape[0], 256, 256, 3)
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			with torch.no_grad():
				out = renderer.render_batch(batch_poses, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
			# batch_image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8) 
			batch_image = out["image"].detach().cpu()
			vis_img[j*chunk_size:(j+1)*chunk_size] = batch_image

		l = [{'params': learnable_cam_params, 'lr': 5e-3, "name": "cam_params"}]
		optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

		init_lr = optimizer.param_groups[0]['lr']
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			optimizer.param_groups[0]['lr'] = init_lr
			for k in tqdm.tqdm(range(s1_steps)):
				batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
				batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
				out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
				pred_tensor = out["image"]
				valid_mask = (out["alpha"] > 0) & (out["viewcos"] > 0.5)  # (500, 256, 256, 1)

				if k == s1_steps - 1:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					loss_MSE_grid[j*chunk_size:(j+1)*chunk_size] = loss.detach().cpu().numpy()
					loss = loss.mean()

				else:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='mean')

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()

		# Render optimized images for visualization
		# vis_img_optimized = torch.zeros(pose_grid.shape[0], 256, 256, 3)
		# for j in tqdm.tqdm(range(num_batches)):
		# 	batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
		# 	batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
		# 	batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
		# 	batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
		# 	with torch.no_grad():
		# 		out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
		# 	# batch_image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8) 
		# 	batch_image = out["image"].detach().cpu()
		# 	vis_img_optimized[j*chunk_size:(j+1)*chunk_size] = batch_image

		# indices = np.argsort(loss_MSE_grid)
		# padding = (pose_grid.shape[0] // 10 + int(pose_grid.shape[0]%10 > 0)) * 10 - pose_grid.shape[0]
		# grid = vis_img[indices].permute(0, 3, 1, 2).contiguous()
		# padded_gird = torch.cat([grid, torch.ones(padding, 3, 256, 256)], dim=0)
		# padded_gird = padded_gird.view((padding + pose_grid.shape[0]) // 10, 10, 3, 256, 256).permute(2, 0, 3, 1, 4)
		# padded_gird = padded_gird.reshape(3, -1, 2560)
		# output_path = os.path.join(save_path, f'vis1_candidates_{i}.png')
		# save_image(padded_gird, output_path)

		# grid = vis_img_optimized[indices].permute(0, 3, 1, 2).contiguous()
		# padded_gird = torch.cat([grid, torch.ones(padding, 3, 256, 256)], dim=0)
		# padded_gird = padded_gird.view((padding + pose_grid.shape[0]) // 10, 10, 3, 256, 256).permute(2, 0, 3, 1, 4)
		# padded_gird = padded_gird.reshape(3, -1, 2560)
		# output_path = os.path.join(save_path, f'vis1_optimized_candidates_{i}.png')
		# save_image(padded_gird, output_path)

		beta = 0.1
		indices = np.argsort(loss_MSE_grid)[:max(int(loss_MSE_grid.shape[0] * beta), 64)]
		batch_poses = poses[indices]
		batch_residuals = SE3.exp(learnable_cam_params[indices].detach()).as_matrix() # [5760, 4, 4]
		poses = torch.bmm(batch_poses, batch_residuals) # [216, 4, 4]
		poses = poses.repeat(4, 1, 1)

		learnable_cam_params = torch.randn(poses.shape[0], 6) * 1e-1
		learnable_cam_params.requires_grad_()

		optimizer.param_groups = []
		optimizer.add_param_group({'params': learnable_cam_params})

		perspectives = torch.from_numpy(cameras[i].perspective).expand(poses.shape[0], -1, -1)
		loss_MSE_grid = np.zeros(poses.shape[0])

		num_batches = poses.shape[0] // chunk_size + int(poses.shape[0]%chunk_size > 0)
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			optimizer.param_groups[0]['lr'] = 1e-3
			for k in tqdm.tqdm(range(s2_steps)):
				batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
				batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
				out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
				pred_tensor = out["image"]
				valid_mask = (out["alpha"] > 0) & (out["viewcos"] > 0.5)  # (500, 256, 256, 1)
				# batch_image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8) 
				# del batch_pose, batch_perspective

				if k == s2_steps - 1:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					# loss += F.mse_loss(valid_mask, gt_mask_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					loss_MSE_grid[j*chunk_size:(j+1)*chunk_size] = loss.detach().cpu().numpy()
					loss = loss.mean()

				else:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='mean')

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()

		beta = 0.1
		indices = np.argsort(loss_MSE_grid)[:max(int(loss_MSE_grid.shape[0] * beta), 64)]
		batch_poses = poses[indices]
		batch_residuals = SE3.exp(learnable_cam_params[indices].detach()).as_matrix() # [5760, 4, 4]
		poses = torch.bmm(batch_poses, batch_residuals) # [216, 4, 4]
		poses = poses.repeat(4, 1, 1)

		learnable_cam_params = torch.randn(poses.shape[0], 6) * 1e-2
		learnable_cam_params.requires_grad_()

		optimizer.param_groups = []
		optimizer.add_param_group({'params': learnable_cam_params})

		perspectives = torch.from_numpy(cameras[i].perspective).expand(poses.shape[0], -1, -1)
		loss_MSE_grid = np.zeros(poses.shape[0])

		num_batches = poses.shape[0] // chunk_size + int(poses.shape[0]%chunk_size > 0)
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			optimizer.param_groups[0]['lr'] = 1e-3
			for k in tqdm.tqdm(range(s2_steps)):
				batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
				batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
				out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
				pred_tensor = out["image"]
				valid_mask = (out["alpha"] > 0) & (out["viewcos"] > 0.5)  # (500, 256, 256, 1)

				if k == s2_steps - 1:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					# loss += F.mse_loss(valid_mask, gt_mask_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					loss_MSE_grid[j*chunk_size:(j+1)*chunk_size] = loss.detach().cpu().numpy()
					loss = loss.mean()

				else:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='mean')

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()

		pose_grid = poses
		loss_LPIPS_grid = np.zeros(poses.shape[0])

		chunk_size = 64
		gt_tensor = gt_tensor.permute(0, 3, 1, 2).contiguous()
		vis_img_opt = np.zeros((pose_grid.shape[0], 256, 256, 3), dtype=np.uint8)
		num_batches = pose_grid.shape[0] // chunk_size + int(pose_grid.shape[0]%chunk_size > 0)
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
			batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			with torch.no_grad():
				out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
			batch_image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8) 
			vis_img_opt[j*chunk_size:(j+1)*chunk_size] = batch_image

			pred_tensor = out["image"].permute(0, 3, 1, 2).contiguous()
			with torch.no_grad():
				loss_LPIPS_grid[j*chunk_size:(j+1)*chunk_size] = lpips_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1)).squeeze().cpu().numpy()

		# indices_of_smallest = np.argsort(loss_MSE_grid)[:15]
		indices1 = np.argsort(loss_MSE_grid)
		indices2 = np.argsort(loss_LPIPS_grid)

		ranks1 = np.zeros_like(loss_MSE_grid)
		ranks2 = np.zeros_like(loss_LPIPS_grid)

		ranks1[indices1] = np.arange(1, loss_MSE_grid.size + 1)
		ranks2[indices2] = np.arange(1, loss_LPIPS_grid.size + 1)

		total_ranks = ranks1 + ranks2
		indices_of_smallest = np.argsort(total_ranks)[:15]

		index = indices_of_smallest[0]
		residual = SE3.exp(learnable_cam_params[index].detach()).as_matrix() # [5760, 4, 4]
		c2w = poses[index] @ residual
		w2c = torch.inverse(c2w)

		w2c[1:3, :] *= -1 # OpenCV to OpenGL
		w2c[:2, :] *= -1 # PyTorch3D to OpenCV

		data[list(data.keys())[i]]["R"] = w2c[:3, :3].T.tolist()
		data[list(data.keys())[i]]["T"] = w2c[:3, 3].tolist()

		num_frames = 16
		cmap = plt.get_cmap("hot")
		num_rows = 2
		num_cols = 8
		# plt.subplots_adjust(top=0.2)
		figsize = (num_cols * 2, num_rows * 2.4)
		fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
		fig.suptitle(f"Input Image v.s. Top 15 Similar Renderings", fontsize=16, y=0.93)
		plt.subplots_adjust(top=0.9)
		axs = axs.flatten()
		for idx in range(num_rows * num_cols):
			if idx < num_frames:
				if idx == 0:
					axs[idx].imshow(gt_img.reshape(256, 256, 3))
					axs[idx].set_xlabel(f'Input Image', fontsize=10)
				else:
					axs[idx].imshow(vis_img_opt[indices_of_smallest[idx-1]].reshape(256, 256, 3))
					loss_text = f"MSE: {loss_MSE_grid[indices_of_smallest[idx-1]]:.4f}_{int(ranks1[indices_of_smallest[idx-1]]):d}\nLPIPS: {loss_LPIPS_grid[indices_of_smallest[idx-1]]:.4f}_{int(ranks2[indices_of_smallest[idx-1]]):d}"
					axs[idx].text(0.05, 0.95, loss_text, color='black', fontsize=8, 
								  ha='left', va='top', transform=axs[idx].transAxes)
				for s in ["bottom", "top", "left", "right"]:
					if idx == 0:
						axs[idx].spines[s].set_color("green")
					else:
						axs[idx].spines[s].set_color(cmap(0.8 * idx / (num_frames)))
					axs[idx].spines[s].set_linewidth(5)
				axs[idx].set_xticks([])
				axs[idx].set_yticks([])

				# if i >= args.all_views:
				#     axs[i].set_xlabel(f'MSE: {mse_losses[i%args.all_views]:.4f}\nLPIPS: {lpips_losses[i%args.all_views]:.4f}', fontsize=10)
			else:
				axs[i].axis("off")
		plt.tight_layout()

		output_path = os.path.join(save_path, f'vis_{i}_render_and_compare.png')
		plt.savefig(output_path)  # Save the figure to a file
		plt.close(fig)
		print(f"Visualization file written to {output_path}")

	del lpips_loss, renderer, learnable_cam_params
	gc.collect()
	torch.cuda.empty_cache()

	return data


def align_to_mesh(camera_data, mesh_path, save_path, num_views=8):
	parser = argparse.ArgumentParser()
	parser.add_argument('--object', type=str, help="path to mesh (obj, ply, glb, ...)")
	parser.add_argument('--path', type=str, help="path to mesh (obj, ply, glb, ...)")
	parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
	parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
	parser.add_argument('--W', type=int, default=256, help="GUI width")
	parser.add_argument('--H', type=int, default=256, help="GUI height")
	parser.add_argument("--wogui", type=bool, default=True, help="disable all dpg GUI")
	parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
	parser.add_argument("--config", default='configs/navi.yaml', help="path to the yaml config file")
	parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
	parser.add_argument('--fovy', type=float, default=49.1, help="default GUI camera fovy")
	args, extras = parser.parse_known_args()

	# override default config from cli
	opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
	data = camera_data

	opt.mesh = mesh_path
	opt.trainable_texture = False
	renderer = Renderer(opt).to(torch.device("cuda"))

	cameras = [CustomCamera(cam_params) for cam_params in data.values()]
	# cams = [(cam.c2w, cam.perspective, cam.focal_length) for cam in cameras]
	img_paths = [v["filepath"] for k, v in data.items()]
	flags = [int(v["flag"]) for k, v in data.items()]

	s1_steps = 128
	num_hypotheses = 64
	chunk_size = 512
	print("Number of hypotheses:", num_hypotheses)

	for i in tqdm.tqdm(range(num_views)):
		if flags[i]:
			continue

		loss_MSE_grid = np.zeros(num_hypotheses)
		vis_img_opt = torch.zeros(num_hypotheses, 256, 256, 3)
		poses = torch.from_numpy(cameras[i].c2w).expand(num_hypotheses, -1, -1)
		perspectives = torch.from_numpy(cameras[i].perspective).expand(num_hypotheses, -1, -1)

		learnable_cam_params = torch.randn(num_hypotheses, 6) * 1e-3
		learnable_cam_params.requires_grad_()

		img_path = img_paths[i]
		gt_img = Image.open(img_path)
		if gt_img.mode == 'RGBA':
			gt_img = np.asarray(gt_img, dtype=np.uint8).copy()
			gt_mask = (gt_img[..., 3:] > 128).astype(np.float32)
			gt_img[gt_img[:, :, -1] <= 255*0.9] = [255., 255., 255., 255.] # thresholding background
			gt_img = gt_img[:, :, :3]

		gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).cuda() / 255.
		gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).cuda()

		num_batches = num_hypotheses // chunk_size + int(num_hypotheses%chunk_size > 0)

		l = [{'params': learnable_cam_params, 'lr': 5e-3, "name": "cam_params"}]
		optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

		init_lr = optimizer.param_groups[0]['lr']
		for j in tqdm.tqdm(range(num_batches)):
			batch_poses = poses[j*chunk_size:(j+1)*chunk_size]
			batch_perspectives = perspectives[j*chunk_size:(j+1)*chunk_size]
			optimizer.param_groups[0]['lr'] = init_lr
			for k in tqdm.tqdm(range(s1_steps)):
				batch_residuals = SE3.exp(learnable_cam_params[j*chunk_size:(j+1)*chunk_size]).as_matrix() # [5760, 4, 4]
				batch_poses_opt = torch.bmm(batch_poses, batch_residuals)
				out = renderer.render_batch(batch_poses_opt, batch_perspectives, 256, 256, ssaa=1)  # (500, 256, 256, 3)
				pred_tensor = out["image"]

				if k == s1_steps - 1:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					# loss += F.mse_loss(valid_mask, gt_mask_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='none').mean(dim=(1, 2, 3))
					loss_MSE_grid[j*chunk_size:(j+1)*chunk_size] = loss.detach().cpu().numpy()
					batch_image = pred_tensor.detach().cpu()
					vis_img_opt[j*chunk_size:(j+1)*chunk_size] = batch_image
					loss = loss.mean()

				else:
					loss = F.mse_loss(pred_tensor, gt_tensor.expand(pred_tensor.shape[0], -1, -1, -1), reduction='mean')

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()

		indices = np.argsort(loss_MSE_grid)
		residual = SE3.exp(learnable_cam_params[indices[0]].detach()).as_matrix() # [5760, 4, 4]
		c2w = torch.from_numpy(cameras[i].c2w) @ residual
		w2c = torch.inverse(c2w)

		w2c[1:3, :] *= -1 # OpenCV to OpenGL
		w2c[:2, :] *= -1 # PyTorch3D to OpenCV

		data[list(data.keys())[i]]["R"] = w2c[:3, :3].T.tolist()
		data[list(data.keys())[i]]["T"] = w2c[:3, 3].tolist()

		grid = vis_img_opt[indices].permute(0, 3, 1, 2).contiguous()
		grid = grid.view(8, 8, 3, 256, 256).permute(2, 0, 3, 1, 4)
		grid = grid.reshape(3, -1, int(256*8))
		output_path = os.path.join(save_path, f'vis_aligned_candidates_{i}.png')
		save_image(grid, output_path)

	return data

