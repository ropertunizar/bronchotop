<!--
README Template for a Research / Benchmark / Code Release Repository
Replace placeholders like <PROJECT_NAME>, <PAPER_TITLE>, <YEAR>, <HOMEPAGE_URL>, etc.
Remove any sections you do not need.
-->

<h1 align="center">BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization</h1>
<h3 align="center">IROS 2026</h3>

<div align="center">
  <a href="https://github.com/ClaraTmsn" target="_blank">Clara Tomasini</a><sup>1</sup>,
  <a href="https://github.com/anacmurillo" target="_blank">Ana C. Murillo</a><sup>1</sup>,
  <a href="https://github.com/luisriazuelo" target="_blank">Luis Riazuelo</a><sup>1</sup>
</div>

<div align="center">
  <sup>1</sup>Universidad de Zaragoza
</div>

<div align="center">
  <a href="https://sites.google.com/unizar.es/bronchotop"><strong>🌍 Homepage</strong></a> |
  <a href="DATASET_OR_BENCHMARK_URL"><strong>🤗 Dataset</strong></a> |
  <a href="https://arxiv.org/abs/2609.28328"><strong>📝 Paper</strong></a> |
  <a href="DEMO_URL"><strong>🎬 Demo</strong></a> 
</div>

---

## 🔔 News
<!-- - ⭐ 10/2026: Benchmark / dataset released. -->
- 🆕 10/2026: Code released.
- 🥳 09/2026: Paper accepted at IROS.
---

## 📖 Description
**BronchoTop** is the first published framework including both the localization algorithms as well as all the real data used, together with code to generate additional simulations, encouraging and facilitating further developments and benchmarking. The results highlight BronchoTop's potential to enhance procedural safety, efficiency and accessibility in clinical and robotic bronchoscopy.

## 🛠️ Requirements
The software has been tested on **Ubuntu 20.04** and uses [Python](https://www.python.org). **Required 3.X**.

```bash
# Clone the repository
git clone https://github.com/ropertunizar/bronchotop.git
cd bronchotop

# Create a virtual environment
conda env create -f environment_bronchotop.yml
conda activate bronchotop
```
## 🚀 Usage
# 1. Data preparation

Create folder ```datasets``` and place your data in it.
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

# 2. Siamese network checkpoint
   
Download checkpoints for siamese network trained our real and simulated sequences [here](https://drive.google.com/drive/folders/1iMw_HVt2ZtEHfKLwnHundaQIPZCpDgFS?usp=sharing). 

# 3. BronchoTop Inference
Run BronchoTop pipeline using

```python bronchotop_main.py --sequence_path path/to/sequence  --data_type (sim or real)```

Additionally, you can specify start and end frames for the processed sequence, as well as whether to use the final Switch Verification module or not.

Parameters such as similarity threshold can be adjusted directly in ```bronchotop_main.py```.

## 📜 License

This work is released under AGPL-3.0 license.
 
## 📝 Citation
```bibtex
@inproceedings{tomasini2026bronchotop,
  title={BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization},
  author={Tomasini, Clara and Murillo, Ana C. and Riazuelo, Luis},
  booktitle={IROS},
  year={2026}
}
```

## 🙏 Acknowledgements
This work was partially funded by projectT45_23R, by grants PID2022-139615OB-I00, AIA2025-163563-C31, PID2024-159284NB-I00,funded by MCIN/AEI/10.13039/501100011033 and ERDF.

