# SparseAGS

This repository contains the official implementation for **Sparse-view Pose Estimation and Reconstruction via Analysis by Generative Synthesis**. The paper has been accepted to [NeurIPS 2024](https://neurips.cc/Conferences/2024).

### [Project Page](https://qitaozhao.github.io/SparseAGS) | [arXiv (Coming Soon)](https://qitaozhao.github.io/SparseAGS)

### News

- 2024.12.02: Initial code release.

## Introduction

**tl;dr** Given a set of unposed input images, **SparseAGS** jointly infers the corresponding camera poses and underlying 3D, allowing high-fidelity 3D inference in the wild.

**Abstract.** Inferring the 3D structure underlying a set of multi-view images typically requires solving two co-dependent tasks -- accurate 3D reconstruction requires precise camera poses, and predicting camera poses relies on (implicitly or explicitly) modeling the underlying 3D. The classical framework of analysis by synthesis casts this inference as a joint optimization seeking to explain the observed pixels, and recent instantiations learn expressive 3D representations (e.g., Neural Fields) with gradient-descent-based pose refinement of initial pose estimates. However, given a sparse set of observed views, the observations may not provide sufficient direct evidence to obtain complete and accurate 3D. Moreover, large errors in pose estimation may not be easily corrected and can further degrade the inferred 3D. To allow robust 3D reconstruction and pose estimation in this challenging setup, we propose *SparseAGS*, a method that adapts this analysis-by-synthesis approach by: a) including novel-view-synthesis-based generative priors in conjunction with photometric objectives to improve the quality of the inferred 3D, and b) explicitly reasoning about outliers and using a discrete search with a continuous optimization-based strategy to correct them. We validate our framework across real-world and synthetic datasets in combination with several off-the-shelf pose estimation systems as initialization. We find that it significantly improves the base systems' pose accuracy while yielding high-quality 3D reconstructions that outperform the results from current multi-view reconstruction baselines.

![teasert](assets/teaser.gif)

## Install

1. Clone SparseAGS:

```bash
git clone --recursive https://github.com/QitaoZhao/SparseAGS.git
cd SparseAGS
# if you have already cloned sparseags:
# git submodule update --init --recursive
```

2. Create the environment and install packages:

```bash
conda create -n sparseags python=3.9
conda activate sparseags

# enbale nvcc
conda install -c conda-forge cudatoolkit-dev

### torch
# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

### pytorch3D
# CUDA 11.7
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1130.tar.bz2

# CUDA 12.1
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt210.tar.bz2

# liegroups (minor modification to https://github.com/utiasSTARS/liegroups)
pip install ./liegroups

# simple-knn
pip install ./simple-knn

# a modified gaussian splatting from https://github.com/ashawkey/diff-gaussian-rasterization, which enables camera pose optimization
pip install ./diff-gaussian-rasterization-camera 

######################## Make it Submodule ########################
# dust3r
git clone --recursive https://github.com/naver/dust3r

# a modified gaussian splatting on top of https://github.com/ashawkey/diff-gaussian-rasterization, which enables camera pose optimization
git clone --recursive git@github.com:QitaoZhao/diff-gaussian-rasterization-camera.git
pip install ./diff-gaussian-rasterization-camera # Try "sudo apt-get install libglm-dev" if you encounter "fatal error: glm/glm.hpp: No such file or directory"

######################## Remove ########################
# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# torch
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Tested on:

- Ubuntu 20.04 with torch 1.13 & CUDA 11.7 on an A5000 GPU.
- Springdale Linux 8.6 with torch 2.1.0 & CUDA 12.1 on an A5000 GPU.
- Red Hat Enterprise Linux 8.10 with torch 1.13 & CUDA 11.7 on a V100 GPU.

Note: Look at this [issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/993) or try `sudo apt-get install libglm-dev` if you encounter `fatal error: glm/glm.hpp: No such file or directory` when doing `pip install ./diff-gaussian-rasterization-camera`. 

3. Download our 6-DoF Zero123 [checkpoint](https://drive.google.com/file/d/1JJ4wjaJ4IkUERRZYRrlNoQ-tXvftEYJD/view?usp=sharing) and place it in `SparseAGS/checkpoints`.

## Usage

(1) **Gradio Demo** (recommended, where you can upload your own images or use our preprocessed examples interactively):

```bash
python gradio_app.py
```

(2) Use command lines:

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

### sparse-view 3D reconstruction
# here we have some preprocessed examples in 'data/demo', with dust3r pose initialization
# the output will be saved in 'output/demo/{category}'
# valid category-num_views options are {[jordan, 8], [butter, 6], [robot, 8], [eagle, 8]}

# run single 3D reconstruction (w/o outlier removal & correction)
python run.py --category jordan --num_views 8 

# if you find the command above does not give you nice 3D, try enbaling loop-based outlier removal & correction (which takes more time)
python run.py --category jordan --num_views 8 --enable_loop
```

Note: Actually, we include the `eagle` example to showcase how our full method works (we found in our experiments that dust3r gives one bad pose for this example). For other examples, you are supposed to see reasonable 3D with a single 3D reconstruction.

## Tips

* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

## Acknowledgments

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

## Citation

```
@inproceedings{zhao2024sparseags,
  title={Sparse-view Pose Estimation and Reconstruction via Analysis by Generative Synthesis}, 
  author={Qitao Zhao and Shubham Tulsiani},
  booktitle={NeurIPS},
  year={2024}
}
```
