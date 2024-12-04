import os
import json
import copy
import shutil
import argparse
import numpy as np

import sys 
import torch

from sparseags.render_utils.util import render_and_compare, align_to_mesh
from sparseags.visual_utils import vis_output


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def do_reconstruction(cur_dir, out_dir, args, force_all_views=False):
    # first stage
    if not os.path.exists(os.path.join(cur_dir, f'{args.category}_mesh.{args.mesh_format}')):
        os.system(f'python sparseags/main_stage1.py '
                  f'--config configs/{args.config} '
                  f'camera_path={cur_dir}/cameras.json ' 
                  f'outdir=./ '
                  f'save_path={cur_dir}/{args.category} '
                  f'opt_cam=1 sh_degree=3 '
                  f'num_pts={args.num_pts} '
                  f'all_views={force_all_views} ' 
                  f'mesh_format={args.mesh_format}')

    # second stage
    if not os.path.exists(os.path.join(cur_dir, f'{args.category}.{args.mesh_format}')):
        os.system(f'python sparseags/main_stage2.py '
                  f'--config configs/{args.config} '
                  f'camera_path={cur_dir}/cameras_updated.json ' 
                  f'outdir=./ '
                  f'save_path={cur_dir}/{args.category} '
                  f'all_views={force_all_views} '
                  f'mesh_format={args.mesh_format}')

    # export video
    mesh_path = os.path.join(cur_dir, f'{args.category}.{args.mesh_format}')
    if not os.path.exists(os.path.join(cur_dir, f'{args.category}.mp4')):
        os.system(f'python -m kiui.render {mesh_path} '
                  f'--save_video {cur_dir}/{args.category}.mp4 '
                  f'--wogui '
                  f'--elevation -30')


def read_updated_cameras(cur_dir): 
    camera_path = os.path.join(cur_dir, 'cameras_updated.json')
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)

    return camera_data


def save_updated_cameras(camera_data, cur_dir): 
    camera_path = os.path.join(cur_dir, 'cameras.json')
    with open(camera_path, 'w') as f:
        json.dump(camera_data, f, indent=4)


def save_metrics(metrics_track, out_dir): 
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_track, f, indent=4)


def main(args):
    CATEGORY = args.category
    NUM_VIEWS = args.num_views
    src_dir = os.path.join('data/demo', CATEGORY)
    out_dir = os.path.join(args.output, CATEGORY)
    os.makedirs(out_dir, exist_ok=True)
    source_camera_path = os.path.join(src_dir, "cameras.json")

    print(f'======== processing {src_dir} ========')
    if not os.path.exists(source_camera_path):
        print(f'{source_camera_path} is missing!')
        sys.exit()

    stop = 0  # the flag for stopping the loop
    stop_with_lower_quality = 0  # the flag for stopping as the reconstruction quality drops
    stop_with_full_iters = 0  # the flag for stopping as we reach the maximum loop number

    cnt = 0
    # pretty empirical choice of maximum iteration number
    if NUM_VIEWS <= 5:
        MAX_CNT = 1
    elif NUM_VIEWS < 8:
        MAX_CNT = NUM_VIEWS - 4
    else:
        MAX_CNT = 4

    if not args.enable_loop:
        MAX_CNT = 0

    THRESHOLD_LPIPS = 0.05
    metrics_track = {}
    cur_dir = os.path.join(out_dir, f'round_{cnt}')
    os.makedirs(cur_dir, exist_ok=True)
    shutil.copy2(source_camera_path, os.path.join(cur_dir, 'cameras.json'))

    while not stop:
        metrics_track[cnt] = []

        do_reconstruction(cur_dir, out_dir, args)
        camera_data = read_updated_cameras(cur_dir)

        lpips_losses, mse_losses = vis_output(
            camera_data, 
            mesh_path=os.path.join(cur_dir, f'{CATEGORY}.{args.mesh_format}'), 
            save_path=os.path.join(cur_dir, 'vis.png'), 
            num_views=NUM_VIEWS
        )
        flags = np.array([int(v["flag"]) for k, v in camera_data.items()])

        mean_lpips = np.sum(lpips_losses * flags) / flags.sum()
        mean_mse = np.sum(mse_losses * flags) / flags.sum()
        metrics_track[cnt].append(mean_lpips)

        # 0: go to the next iter
        # 1-MAX_CNT: stop if no improvement compared to the last iter
        if cnt != 0 and args.enable_loop:
            last_lpips = mean_lpips_wo_max
            diff_lpips = abs(last_lpips - mean_lpips)
            if mean_lpips > last_lpips or diff_lpips < THRESHOLD_LPIPS: 
                stop_with_lower_quality = 1

            if cnt >= MAX_CNT:
                stop_with_full_iters = 1

        if stop_with_full_iters or stop_with_lower_quality:
            stop = 1
            cnt_to_stop = cnt - 1 if stop_with_lower_quality else cnt
            camera_path_to_be_copied = os.path.join(out_dir, f'round_{cnt_to_stop}', 'cameras_updated.json')
            shutil.copy2(camera_path_to_be_copied, os.path.join(out_dir, 'cameras_outlier_removal.json'))
            save_metrics(metrics_track, out_dir)

        elif not args.enable_loop:
            stop = 1
            cnt_to_stop = 0
            save_metrics(metrics_track, out_dir)

        # should not stop, go to the next round
        else:
            max_lpips_value = -float('inf')
            max_index = -1

            for i in range(NUM_VIEWS):
                if flags[i] == 1 and lpips_losses[i] > max_lpips_value:
                    max_lpips_value = lpips_losses[i]
                    max_index = i

            flags[max_index] = 0
            mean_lpips_wo_max = np.sum(lpips_losses * flags) / flags.sum()
            metrics_track[cnt].append(mean_lpips_wo_max)

            assert camera_data[list(camera_data.keys())[max_index]]["flag"] == 1
            camera_data[list(camera_data.keys())[max_index]]["flag"] = 0

            cnt += 1
            cur_dir = os.path.join(out_dir, f'round_{cnt}')
            os.makedirs(cur_dir, exist_ok=True)

            # copy-paste the camera poses for next iter
            save_updated_cameras(camera_data, cur_dir)

    if cnt_to_stop == 0:
        pass

    else:
        """If we identified outliers, do render-and-compare to correct them"""
        camera_path_outlier_removal = os.path.join(out_dir, 'cameras_outlier_removal.json')
        assert os.path.exists(camera_path_outlier_removal)
        with open(camera_path_outlier_removal, 'r') as f:
            camera_data_outlier_removal = json.load(f)

        camera_path_render_and_compare = os.path.join(out_dir, 'cameras_render_and_compare.json')
        if not os.path.exists(camera_path_render_and_compare):
            mesh_path = os.path.join(out_dir, f'round_{cnt_to_stop}', f'{CATEGORY}.{args.mesh_format}')
            camera_data_render_and_compare = render_and_compare(copy.deepcopy(camera_data_outlier_removal), mesh_path, out_dir, num_views=NUM_VIEWS)

            with open(camera_path_render_and_compare, 'w') as f:
                json.dump(camera_data_render_and_compare, f, indent=4)

        # (1) check the recovered cameras from render-and-compare: we do reconstruction to align everything together
        cur_dir = os.path.join(out_dir, f'check_recovered_poses')
        os.makedirs(cur_dir, exist_ok=True)
        shutil.copy2(camera_path_render_and_compare, os.path.join(cur_dir, 'cameras.json'))
        do_reconstruction(cur_dir, out_dir, args, force_all_views=True)

        camera_data = read_updated_cameras(cur_dir)

        lpips_losses, mse_losses = vis_output(
            camera_data, 
            mesh_path=os.path.join(cur_dir, f'{CATEGORY}.{args.mesh_format}'), 
            save_path=os.path.join(cur_dir, 'vis.png'), 
            num_views=args.num_views
        )

        # (2) re-consider initial cameras: we fix the updated inliers while aligning the outliers to the 3D from inliers
        cur_dir = os.path.join(out_dir, f'reconsider_init_poses')
        os.makedirs(cur_dir, exist_ok=True)
        mesh_path = os.path.join(out_dir, f'round_{cnt_to_stop}', f'{CATEGORY}.{args.mesh_format}')
        camera_data_aligned = align_to_mesh(camera_data_outlier_removal, mesh_path, cur_dir, num_views=NUM_VIEWS)
        save_updated_cameras(camera_data_aligned, cur_dir)

        # we do reconstruction to align everything together
        do_reconstruction(cur_dir, out_dir, args, force_all_views=True)
        camera_data_init = read_updated_cameras(cur_dir)

        lpips_losses_init, mse_losses_init = vis_output(
            camera_data_init, 
            mesh_path=os.path.join(cur_dir, f'{CATEGORY}.{args.mesh_format}'), 
            save_path=os.path.join(cur_dir, 'vis.png'), 
            num_views=NUM_VIEWS
        )

        flags_sum = np.array([int(v["flag"]) for k, v in camera_data_outlier_removal.items()]).sum()
        cnt_valid_cameras = 0
        keep_init_poses = False
        if lpips_losses.mean() > lpips_losses_init.mean():
            keep_init_poses = True  # Keep optimized initial poses

        else:
            for idx, (k, v) in enumerate(camera_data_outlier_removal.items()):
                if int(v["flag"]) == 1:
                    continue

                if lpips_losses[idx] < lpips_losses_init[idx] and mse_losses[idx] < mse_losses_init[idx]:
                    cnt_valid_cameras += 1
                    # camera_data_init_updated[k] = camera_data[k] # replace the initial poses with recovered poses

            if cnt_valid_cameras + flags_sum == NUM_VIEWS:
                keep_init_poses = False
            else:
                keep_init_poses = True

        output_path = os.path.join(out_dir, 'cameras_final.json') 
        if keep_init_poses:
            print("Keep the (optimized) initial camera poses.")
            with open(output_path.replace(".json", "_init.json"), 'w') as f:
                json.dump(camera_data_init, f, indent=4)
        else:
            print("Replace the initial cameras with the recovered ones!")
            with open(output_path.replace(".json", "_recovered.json"), 'w') as f:
                json.dump(camera_data, f, indent=4)


if __name__ == "__main__":

    seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/demo', type=str, help='Directory where obj files will be saved')
    parser.add_argument('--category', default='jordan', type=str, help='Directory where obj files will be saved')
    parser.add_argument('--num_pts', default=25000, type=int, help='Number of points at initialization')
    parser.add_argument('--num_views', default=8, type=int, help='Number of input images')
    parser.add_argument('--mesh_format', default='obj', type=str, help='Format of output mesh')
    parser.add_argument('--enable_loop', action='store_true', help='Enable the loop-based strategy to detect and correct outliers')
    parser.add_argument('--config', default='navi.yaml', type=str, help='Path to config file')
    args = parser.parse_args()

    main(args)


