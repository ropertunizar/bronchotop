# BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization

**Authors:** Clara Tomasini, Ana C. Murillo, Luis Riazuelo

### Related Publications:
Tomasini, Clara, Ana C. Murillo and Luis Riazuelo. "BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization" IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026. [**PDF**](https://arxiv.org/abs/2609.28328)

# License
This repository is released under AGPLv3 license.

# Prerequisites
The software has been tested on **Ubuntu 20.04** and uses [Python](https://www.python.org). **Required 3.X**.

# Proposed pipeline
![fig1](https://github.com/user-attachments/assets/2b559a5d-1ac8-4434-a5a5-27c5bc655b84)

# Usage

## Data
Create folder datasets and place your data in it.
Each sequence needs to be stored in its own folder as images and should be organised as follows:

```
 datasets
    ├── sequence1
    │       ├── Class1
    │       │   ├── img1.png
    │       │   ├── img2.png
    │       │   └── img3.png
    │       └── Class2
    │           ├── img4.png
    │           ├── img5.png
    │           └── img6.png
    └── sequence2
```


BronchoTop dataset can be downloaded from XXXX (TO-DO).

## Create environment
Create BronchoTop environment from file using


```conda env create -f environment_bronchotop.yml```


then activate environment

```conda activate bronchotop```

## Siamese network checkpoint
Download checkpoints for siamese network trained our real and simulated sequences [here](https://drive.google.com/drive/folders/1iMw_HVt2ZtEHfKLwnHundaQIPZCpDgFS?usp=sharing).

## BronchoTop Inference
Run BronchoTop pipeline using

```python bronchotop_main.py --sequence_path path/to/sequence  --data_type (sim or real)```

Additionally, you can specify start and end frames for the processed sequence, as well as whether to use the final Switch Verification module or not.

Parameters such as similarity threshold can be adjusted directly in ```bronchotop_main.py```.
