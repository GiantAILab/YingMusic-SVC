# YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases

---

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-YingMusic--SVC-blue)](tech_report/YingMusic-SVC_tech_report.pdf)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--SVC-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC)
[![ModelScope](https://img.shields.io/badge/🔮%20ModelScope-YingMusic--SVC-purple)](https://www.modelscope.cn/models/giantailab/YingMusic-SVC/)
[![Demo Page](https://img.shields.io/badge/🎧%20Demo%20Page-YingMusic--SVC-brightgreen)](https://giantailab.github.io/YingMusic-SVC)



[//]: # ([![Hugging Face]&#40;https://img.shields.io/badge/Demo-YingMusic--SVC-green&#41;]&#40;&#41;)

</div>

---

## Overview ✨

<p align="center">
  <img src="figs/head.jpeg" width="720" alt="pipeline">
</p>

Singing voice conversion (SVC) aims to render the target singer’s timbre while preserving melody and lyrics. However, existing zero-shot SVC systems remain fragile in real songs due to harmony interference, F0 errors, and the lack of inductive biases for singing.
We propose **YingMusic-SVC**, a robust zero-shot framework that unifies continuous pre-training, robust supervised fine-tuning, and Flow-GRPO reinforcement learning. Our model introduces a singing-trained RVC timbre shifter for timbre–content disentanglement, an F0-aware timbre adaptor for dynamic vocal expression, and an energy-balanced rectified flow matching loss to enhance high-frequency fidelity.
Experiments on a graded multi-track benchmark show that YingMusic-SVC achieves consistent improvements over strong open-source baselines in timbre similarity, intelligibility, and perceptual naturalness—especially under accompanied and harmony-contaminated conditions—demonstrating its effectiveness for real-world SVC deployment.

### 🔧 Key Features  
- **Three‑Stage Training Pipeline**  
  - **CPT**: Continuous Pre-Training with singing‑trained modules  
  - **SFT**: Robust Supervised Fine-Tuning with *F0 perturbation* & *harmony augmentation*  
  - **RL (Flow‑GRPO)**: Multi-reward reinforcement learning for perceptual quality  

- **Singing-Specific Inductive Biases**  
  - 🎼 **RVC-based Timbre Shifter** (trained on 120 singers)  
  - 🎚️ **F0‑Aware Fine-Grained Timbre Adaptor**  
  - 🔊 **Energy-balanced Flow Matching Loss** (enhanced high-frequency details)

---


<p align="center">
  <img src="figs/svc_main.jpg" width="720" alt="pipeline">
</p>

---

## News & Updates 🗞️
- **2025-11-26**: Released our accompany separator inference CLI and model ckpt
- **2025-11-26**: Released gradio app for easy try    
- **2025-11-25**: Released technical report
- **2025-11-25**: Initial YingMusic-SVC inference CLI  
- **2025-11-25**: Released model checkpoint  
- **2025-11-25**: Released multi-track benchmark

---

## Installation 🛠️

```bash
git clone https://github.com/GiantAILab/YingMusic-SVC.git
cd YingMusic-SVC

conda create -n ymsvc python=3.10
conda activate ymsvc
pip install -r requirements.txt

# install ffmpeg & sox
sudo apt update
sudo apt install -y sox libsox-fmt-all
sudo apt install -y ffmpeg
```

---

## Quick Start 🚀

### 1. **accompany separation**


```bash

cd accom_separation
bash infer.sh

```

### 2. **SVC Inference**

```bash
bash my_infer.sh
```

### 3. **Gradio APP**

```bash
python gradio_app.py
```
---

## Benchmark Datasets 📚

We provide a **graded difficulty benchmark**, derived from 100+ multi-track studio songs:

[🤗 Download](https://huggingface.co/datasets/GiantAILab/YingMusic-SVC_Difficulty-Graded_Benchmark)
[🔮 Download](https://www.modelscope.cn/datasets/giantailab/YingMusic-SVC_Difficulty-Graded_Benchmark)

| Level | Description |
|-------|-------------|
| **GT Leading** | Clean studio lead vocals |
| **Mix Vocal** | Lead + harmony contamination |
| **Ours Leading** | Extracted via our Band RoFormer separator |


---

## Pretrained Models 🧪

| Model              | Description                     | Link |
|--------------------|--------------------------------|------|
| **YingMusic-SVC-full** | RL-enhanced final model         | [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--SVC--Full-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt) |
| **our BR separator** | Our accompany separation model | [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-BR--separator-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/bs_roformer.ckpt) |


---

## Development Roadmap & TODO 🗺️
- [x] our stem-separator inference CLI & model ckpt
- [x] develop gradio app for YingMusic-SVC
- [ ] benchmark one-click eval script

---

## Acknowledgements 🙏  

This project is built upon:

- [Seed-VC](https://github.com/Plachtaa/seed-vc)

[//]: # (- BigVGAN Vocoder  )

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # ()
[//]: # (## Citation 🧾)

[//]: # ()
[//]: # (If you use YingMusic‑SVC for research, please cite:)

[//]: # ()
[//]: # (```)

[//]: # (@article{yingmusic_svc_2025,)

[//]: # (  title={YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases},)

[//]: # (  author={Chen, Gongyu and Zhang, Xiaoyu and Weng, Zhenqiang and Zheng, Junjie and Shen, Da and Ding, Chaofan and Zhang, Wei-Qiang and Chen, Zihao},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

---

## License 📝  
Our code is released under MIT License.
