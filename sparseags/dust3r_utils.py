import torch
from pytorch3d.renderer import PerspectiveCameras

import sys 
sys.path.append('./')
from sparseags.cam_utils import normalize_cameras_with_up_axis

sys.path[0] = sys.path[0] + '/dust3r'
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def infer_dust3r(dust3r_model, file_names, device='cuda'):
	batch_size = 1
	schedule = 'cosine'
	lr = 0.01
	niter = 300

	images = load_images(file_names, size=224)
	pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
	output = inference(pairs, dust3r_model, device, batch_size=batch_size)

	scene = global_aligner(output, optimize_pp=True, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
	loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

	# retrieve useful values from scene:
	imgs = scene.imgs 
	cams2world = scene.get_im_poses() 
	w2c = torch.linalg.inv(cams2world)
	pps = scene.get_principal_points() * 256 / 224
	focals = scene.get_focals() * 256 / 224

	w2c[:, :2] *= -1  # OpenCV to PyTorch3D
	Rs = w2c[:, :3, :3].transpose(1, 2)
	Ts = w2c[:, :3, 3]

	cameras = PerspectiveCameras(
		focal_length=focals,
		principal_point=pps,
		in_ndc=False,
		R=Rs,
		T=Ts,
	)
	normalized_cameras, _, _, _, _, needs_checking = normalize_cameras_with_up_axis(cameras, None, in_ndc=False)

	if normalized_cameras is None:
		print("It seems something wrong...")
		return 0

	data = {}
	base_names = [file_name.split('/')[-1].split('.')[0] for file_name in file_names]
	file_names = [file_name.replace('source', 'processed').replace('.png', '_rgba.png') for file_name in file_names]

	for idx, base_name in enumerate(base_names):
		data[base_name] = {}
		data[base_name]["R"] = normalized_cameras.R[idx].cpu().tolist()
		data[base_name]["T"] = normalized_cameras.T[idx].cpu().tolist()
		data[base_name]["needs_checking"] = needs_checking
		data[base_name]["principal_point"] = normalized_cameras.principal_point[idx].cpu().tolist()
		data[base_name]["focal_length"] = normalized_cameras.focal_length[idx].cpu().tolist()
		data[base_name]["flag"] = 1
		data[base_name]["filepath"] = file_names[idx]

	return data