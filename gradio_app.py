import sys
import gradio as gr
import os
import shutil
import json
import argparse
from PIL import Image
import subprocess
from sparseags.dust3r_utils import infer_dust3r
from run import main
import functools

sys.path[0] = sys.path[0] + '/dust3r'
from dust3r.model import AsymmetricCroCo3DStereo


def info_fn():
    gr.Info("Data preprocessing done!")


def get_select_index(evt: gr.SelectData):
	index = evt.index
	cate_list = ['toy', 'butter', 'robot', 'jordan', 'eagle']
	args.num_views = len(examples_full[index][0])
	args.category = cate_list[index]

	return examples_full[index][0], examples_full[index][0]


# check if there is a picture uploaded or selected
def check_img_input(control_image):
	if control_image is None:
		raise gr.Error("Please select or upload an input image")


def preprocess(args, dust3r_model, image_block: list):
	if os.path.exists('data/demo/custom'):
		shutil.rmtree('data/demo/custom')

	if os.path.exists('output/demo/custom'):
		shutil.rmtree('output/demo/custom')

	os.makedirs('data/demo/custom/source')
	os.makedirs('data/demo/custom/processed')

	file_names = []

	for file_path in image_block:
		file_name = file_path.split("/")[-1]
		img_pil = Image.open(file_path)

		# save image to a designated path
		try:
			img_pil.save(os.path.join('data/demo/custom', file_name))
		except OSError:
			img_pil = img_pil.convert('RGB')
			img_pil.save(os.path.join('data/demo/custom', file_name))

		file_names.append(os.path.join('data/demo/custom/source', file_name.split('.')[0] + '.png'))

		# crop and resize image
		print(f"python process.py {os.path.join('data/demo/custom', file_name)}")
		subprocess.run(f"python process.py {os.path.join('data/demo/custom', file_name)}", shell=True)

	# predict initial camera poses from dust3r
	camera_data = infer_dust3r(dust3r_model, file_names)
	with open(os.path.join('data/demo/custom', 'cameras.json'), "w") as f:
		json.dump(camera_data, f)

	args.num_views = len(file_names)
	args.category = "custom"

	processed_image_block = []
	for file_path in image_block:
		out_base = os.path.basename(file_path).split('.')[0]
		out_rgba = os.path.join('data/demo/custom/processed', out_base + '_rgba.png')
		processed_image_block.append(out_rgba)

	return processed_image_block


def run_single_reconstruction(image_block: list):
	args.enable_loop = False
	main(args)

	return f'output/demo/{args.category}/round_0/{args.category}.glb'


def run_full_reconstruction(image_block: list):
	args.enable_loop = True
	main(args)

	if os.path.exists(f'output/demo/{args.category}/cameras_final_recovered.json'):
		return f'output/demo/{args.category}/check_recovered_poses/{args.category}.glb'
	elif os.path.exists(f'output/demo/{args.category}/cameras_final_init.json'):
		return f'output/demo/{args.category}/reconsider_init_poses/{args.category}.glb'
	else:
		return f'output/demo/{args.category}/round_1/{args.category}.glb'


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', default='output/demo', type=str, help='Directory where obj files will be saved')
	parser.add_argument('--category', default='jordan', type=str, help='Directory where obj files will be saved')
	parser.add_argument('--num_pts', default=25000, type=int, help='Number of points at initialization')
	parser.add_argument('--num_views', default=8, type=int, help='Number of input images')
	parser.add_argument('--mesh_format', default='glb', type=str, help='Format of output mesh')
	parser.add_argument('--enable_loop', default=True, help='Enable the loop-based strategy to detect and correct outliers')
	parser.add_argument('--config', default='navi.yaml', type=str, help='Path to config file')
	args = parser.parse_args()

	_TITLE = '''Sparse-view Pose Estimation and Reconstruction via Analysis by Generative Synthesis'''

	# <a style="display:inline-block; margin-left: .5em" href="https://openreview.net/pdf?id=wgpmDyJgsg"><img src="https://img.shields.io/badge/2309.16653-f9f7f7?logo=data:image/png;base64,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"></a>

	_DESCRIPTION = '''
	<div>
	<a style="display:inline-block" href="https://qitaozhao.github.io/SparseAGS"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
	<a style="display:inline-block; margin-left: .5em" href='https://github.com/QitaoZhao/SparseAGS'><img src='https://img.shields.io/github/stars/QitaoZhao/SparseAGS?style=social'/></a>
	</div>
	Given a set of unposed input images, SparseAGS jointly infers the corresponding camera poses and underlying 3D, allowing high-fidelity 3D inference in the wild. 
	'''
	_IMG_USER_GUIDE = "Once the preprocessed images appear, click **Run Single 3D Reconstruction**. If the 3D reconstruction looks bad, try **Outlier Removal & Correction** to handle outlier camera poses using the full method."

	# load images in 'data/demo' folder as examples
	examples_full = []

	for example in ['toy', 'butter', 'robot', 'jordan', 'eagle']:
		example_folder = os.path.join(os.path.dirname(__file__), 'data/demo', example, 'processed')
		example_fns = os.listdir(example_folder)
		example_fns.sort()
		examples = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]
		examples_full.append([examples])

	dust3r_model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_224_linear').to('cuda')
	print("Loaded DUSt3R model!")

	preprocess = functools.partial(preprocess, args, dust3r_model)
	# get_select_index = functools.partial(get_select_index, args)

	# Compose demo layout & data flow
	with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
		with gr.Row():
			with gr.Column(scale=1):
				gr.Markdown('# ' + _TITLE)
		gr.Markdown(_DESCRIPTION)

		# Image-to-3D
		with gr.Row(variant='panel'):
			with gr.Column(scale=5):
				image_block = gr.Files(file_count="multiple")

				preprocess_btn = gr.Button("Preprocess Images")

				gr.Markdown(
					"You can run our model by either: (1) **Uploading images** above and clicking **Preprocess Images** to initialize poses with DUSt3R.  \
					(2) Selecting a **preprocessed example** below to skip preprocessing.")

				gallery = gr.Gallery(
					value=[example[0][0] for example in examples_full], label="Examples", show_label=True, elem_id="gallery"
				, columns=[5], rows=[1], object_fit="contain", height="256", preview=None, allow_preview=None)

				preprocessed_data = gr.Gallery(
 					label="Preprocessed images", show_label=True, elem_id="gallery"
				, columns=[4], rows=[2], object_fit="contain", height="256", preview=None, allow_preview=None)

				with gr.Row(variant='panel'):
					run_single_btn = gr.Button("Run Single 3D Reconstruction")
					outlier_detect_btn = gr.Button("Outlier Removal & Correction")
				img_guide_text = gr.Markdown(_IMG_USER_GUIDE, visible=True)

			with gr.Column(scale=5):
				obj_single_recon = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Single Reconstruction)")
				obj_outlier_detect = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Full Method, w/ Outlier Removal & Correction)")

			# Select an preprocessed example
			gallery.select(get_select_index, None, outputs=[image_block, preprocessed_data])

			# Upload you own images and run preprocessing
			preprocess_btn.click(preprocess, inputs=[image_block], outputs=[preprocessed_data], queue=False, show_progress='full').success(info_fn, None, None)

			# Do single 3D reconstruction
			run_single_btn.click(check_img_input, inputs=[image_block], queue=False).success(run_single_reconstruction,
																						  inputs=[image_block],
																								  # preprocess_chk],
																								  # elevation_slider],
																						  outputs=[obj_single_recon])

			# Do loop-based outlier removal & correction                                                                                  
			outlier_detect_btn.click(check_img_input, inputs=[image_block], queue=False).success(run_full_reconstruction,
																						  inputs=[image_block],
																								  # preprocess_chk],
																								  # elevation_slider],
																						  outputs=[obj_outlier_detect])

	demo.queue().launch(share=True)