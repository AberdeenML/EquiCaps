<h1 align="center">
 3DIEBench-T: 3D Invariant Equivariant Benchmark-Translated
</h1>

The dataset introduced in this work is publicly available on [Hugging Face](https://huggingface.co/datasets/athenacon/3DIEBench-T).

We provide the code to reproduce it from scratch in this directory.
<p align="center">
<img src="/misc/samples_3DIEBench-T.png" alt="Samples 3DIEBench-T" width="700"/>
</p>

## Dataset Summary
3DIEBench-T (3D Invariant Equivariant Benchmark–Translated) is a synthetic vision benchmark designed to evaluate invariant and equivariant representations under 3D transformations. Extending the original [3DIEBench](https://github.com/facebookresearch/SIE) rotation benchmark, 3DIEBench-T incorporates translations to the existing rotations by controlling the scene and object parameters for 3D rotation and translation prediction (equivariance) while retaining sufficient complexity for classification (invariance). Moving from SO(3) transformations to the more general SE(3) group, 3DIEBench-T increases the difficulty of both invariance and equivariance tasks, enabling the evaluation of representation methods under more realistic, simultaneous, multi-geometric settings. 

## Dataset Description
We adhere to the original [3DIEBench](https://github.com/facebookresearch/SIE) data generation protocol, ensuring any observed performance differences are attributed to the inclusion of translations rather than broader dataset modifications.
We use 52,472 3D object instances spanning 55 classes from [ShapeNetCoreV2](https://arxiv.org/abs/1512.03012), originally sourced from [3D Warehouse](https://3dwarehouse.sketchup.com/).
For each instance, we generate 50 uniformly sampled views within specified ranges via [Blender-Proc](https://joss.theoj.org/papers/10.21105/joss.04901), yielding 2,623,600 images of size 256x256. Note that the 55 classes are not balanced.

## Dataset Structure
The data are structured as indicated below.
```
├── SYNSET_1                       
│   ├── OBJ_ID_1_1  
|   |   ├─ image_0.jpg
|   |   ├─ latent_0.npy
|   |   :
|   |   :
|   |   ├─ image_49.jpg
|   |   └─ latent_49.npy        
|   :
|   :       
│   └── OBJ_ID_1_N        
|   |   ├─ image_0.jpg
|   |   ├─ latent_0.npy
|   |   :
|   |   :
|   |   ├─ image_49.jpg
|   |   └─ latent_49.npy                
:
:              
├── SYNSET_55                       
│   ├── OBJ_ID_55_1  
|   |   ├─ image_0.jpg
|   |   ├─ latent_0.npy
|   |   :
|   |   :
|   |   ├─ image_49.jpg
|   |   └─ latent_49.npy        
|   :
|   :       
│   └── OBJ_ID_55_M        
|   |   ├─ image_0.jpg
|   |   ├─ latent_0.npy
|   |   :
|   |   :
|   |   ├─ image_49.jpg
|   |   └─ latent_49.npy              
└── LICENSE 
```

As shown above, we provide the latent information for each generated image to facilitate downstream tasks, in the order shown below. Tait–Bryan angles are used to define extrinsic object rotations, and the light’s position is specified using spherical coordinates.
<ul>
  <li>Rotation X $\in [-\frac{\pi}{2},\frac{\pi}{2}] $</li>
  <li>Rotation Y $\in [-\frac{\pi}{2},\frac{\pi}{2}] $</li>
  <li>Rotation Z $\in [-\frac{\pi}{2},\frac{\pi}{2}] $</li>
  <li>Floor hue $\in [0,1] $</li>
  <li>Light $\theta \in [0,\frac{\pi}{4}] $</li>
  <li>Light $\phi \in [0,2\pi] $</li>
  <li>Light hue $\in [0,1] $</li>
  <li>Translation X $\in [-0.5,0.5] $</li>
  <li>Translation Y $\in [-0.5,0.5] $</li>
  <li>Translation Z $\in [-0.5,0.5] $</li>
</ul>

## Data Splits
The 3DIEBench-T dataset has 2 splits: _train_ and _validation_: 80\% of the objects form the training set, and the remaining 20\%, sampled from the same transformation distribution, form the validation set.
We indicate the statistics below.
The splits are available in the [data](/data/) directory.

| Dataset Split | Number of Objects          | Number of Object Instances (images) (objects &times; views ) |
| ------------- | ---------------------------|--------------------------------------------------------------|
| Train         | 41,920                      | 2,096,000    |
| Validation    | 10,552                      | 527,600      |

## Dataset Reproducibility
To reproduce the dataset from scratch, you can run the 12 scripts we provide in the [scripts](./scripts/) directory. We also provide a [script](./create_scripts.py) that generates these files to run it with different arguments. 

The dataset generation requires approximately 44 hours on 13 NVIDIA A100 80GB GPUs if you run them in parallel, and all random seeds are fixed to enable reproducibility. Alternatively, you can run the same steps sequentially in a single script, albeit with a proportionally longer runtime.

## Contact
If you need help reproducing or using the dataset, please feel free to open an issue or contact the paper's corresponding author directly.

## Acknowledgements 
The dataset generation is built on the [SIE](https://github.com/facebookresearch/SIE) repository.

## Citation
If you use this data, or build on it, please cite the main paper.
```
@article{konstantinou2025equicaps,
  title={EquiCaps: Predictor-Free Pose-Aware Pre-Trained Capsule Networks},
  author={Konstantinou, Athinoulla and Leontidis, Georgios and Thota, Mamatha and Durrant, Aiden},
  journal={arXiv preprint arXiv:2506.09895},
  year={2025}
}
```
