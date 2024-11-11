# 3D Gaussian Splatting -- Compression

## Render strategy

The render strategy is to combine the 3D Gaussian models in one scene with different resolution to render the final image.

- First, use the original training script with different densify_grad_threshold:
- ```python
    python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to save model> --densify_grad_threshold 0.1
- Second, use the render_multiModel.py script to render the final image:
- ```python
    python render_multiModel.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model> --model_paths <path to other trained models that combined with>
  
Command line arguments for render_multiModel.py:
- --model_paths: path to other trained models that combined with
- --strategy: render strategy, default is 'dist', other options are 'fov' and 'distFov'
- --render_image: save rendered image


## LineGS: 3D Line Segment Representation on 3D Gaussian Splatting

LineGS is a 3D line segment representation on 3D Gaussian splatting. This method is a post-processing step after training the 3D Gaussian splatting model.

This method follows the steps below:
- train the 3D Gaussian splatting model with the original training script:
- ```python
    python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to save model>
- use Line3D++ or other 3D line segment detection methods to generate initial 3D line segments.
- use line3d.py to implement LineGS, and the result will be saved in the folder named the same as the folder that contains 3D line segments with prefix '_test'.
- ```python
    python line3d.py -m <path to trained model> -s <path that contains 3D line segments> --baseline 2
- The format of the input 3D line segments can be: .obj, .stl,
- THe format of the output 3D line segments is .stl

Command line arguments for line3d.py:
- --baseline: baseline method, default is 2, other options are 1 and 3:
  - 1: use the dataset for training the 3D Gaussian splatting model to generate trick colmap data folder, which contains images.txt, points3D.txt, and cameras.txt if the original data is in nerf_synthetic format.
  - 3: implement LineGS in different scenes, with different values of parameters in LineGS, and plot the results in figure.
- -s:
  - if the baseline is 1, the path to the dataset for training the 3D Gaussian splatting model
  - if the baseline is 2, the path to the dataset that contains the initial 3D line segments
  - if the baseline is 3, the path to the dataset for implementing LineGS, the data in the folder should contain the initial 3D line segments.

Dataset structure for baseline 3:
```
|- <path to dataset>
    |- <scene name 1>
        |- <colmop data folder>
            |- <3d line segments method>
                |- <name>.obj
                |- <name>.stl
            |- images.txt
            |- points3D.txt
            |- cameras.txt
    |- <scene name 2>
        |- <colmop data folder>
            |- <3d line segments method>
                |- <name>.obj
                |- <name>.stl
            |- images.txt
            |- points3D.txt
            |- cameras.txt
    |- ...
```
