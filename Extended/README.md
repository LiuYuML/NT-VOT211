# Extended Code for IJCV 2026      ![Static Badge](https://img.shields.io/badge/Token_Enhancer-red)![Static Badge](https://img.shields.io/badge/Light_Weight-green)![Static Badge](https://img.shields.io/badge/High_Dimensional_Token_Analysis-blue)


This repository contains the official code and resources for the **extended journal version**:  
**[A Comprehensive Benchmark for Evaluating Night-time Visual Object Tracking](https://doi.org/10.1007/s11263-025-02661-7)**  
(*International Journal of Computer Vision*, 2026)


Yu Liu,  [Arif Mahmood](https://scholar.google.com.sg/citations?user=_e6yGs4AAAAJ&hl=en),  [Muhammad Haris Khan](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en)


## Abstract
>Night-time tracking fails because background and target tokens become indistinguishable in low light. We propose a lightweight Token Enhancer (TE) that separates them in embedding space—trained only on daylight data, no night samples needed. It boosts SOTA trackers on 4 night benchmarks with +0.2M params and +0.15 GFLOPs—the first zero-shot domain adaptation method for VOT.


## Code
In this work, we introduce a novel analytical perspective on night-time tracking failures—namely, the collapse of target-background token separability in low-light conditions—and propose a lightweight, plug-and-play module that can be seamlessly integrated into any existing tracking algorithm. Remarkably, it requires only minimal fine-tuning on standard daylight datasets to achieve significant performance gains in night-time scenarios.

The implementation of the proposed **Token Enhancer (TE)** module is located at:

[Extended/LoRAT_pytracking-main/lib/models/lorat/lorat.py](https://github.com/LiuYuML/NT-VOT211/blob/main/Extended/LoRAT_pytracking-main/lib/models/lorat/lorat.py)

The main results and model checkpoints are stored in:

[Google  Could](https://drive.google.com/drive/folders/1owJkZiW8PPINPXM1ucRhXEKBpEe23s-8?usp=drive_link)

[Bai DU](https://pan.baidu.com/s/1RZgDQs0pIflp6b95rrqHbA?pwd=37xc)


Our code is heavily borrowed from [OSTrack](), [LoRAT](), to run out sript,please check:


[README](https://github.com/LiuYuML/NT-VOT211/edit/main/Extended/LoRAT_pytracking-main)



<details>
<summary><b>Methodology Summary (Extended Only)</b></summary>

We propose a **zero-shot domain adaptation method** for night-time visual object tracking that **requires no nighttime training data**.  

**Core Insight**:  
Under low-light conditions, background tokens become **highly similar to the target token** in the embedding space, causing trackers to drift.

**Method**:  
We introduce a lightweight, plug-and-play **Token Enhancer (TE)** module that:
1. **Reprojects** input tokens into a new space using a small Transformer block.
2. **Differentiates** background tokens from the target token using a location-prior-guided background extractor and a learnable MLP.
3. **Replaces** background tokens with enhanced versions before position estimation.

**Training**:  
- Trained **only on standard daylight datasets** (e.g., GOT-10k) for **11 epochs**.
- Uses a novel **differentiation loss** that pushes background representations away from their original positions in embedding space.

**Key Properties**:
- **Zero-shot**: No night-time data used in training.
- **Lightweight**: Adds only **0.20M parameters** and **0.15 GFLOPs**.
- **Plug-and-play**: Easily integrated into any token-based tracker (e.g., LoRAT, OSTrack).

**Result**:  
Consistently boosts SOTA trackers across **4 night-time benchmarks**, achieving **new SOTA performance** — the **first zero-shot domain adaptation approach in VOT**.

<img width="776" height="405" alt="image" src="https://github.com/user-attachments/assets/a8d9c045-4d02-45f5-8105-4479331809fb" />



</details>


<details>
<summary><b>Qualitative Results</b></summary>

<img width="898" height="785" alt="image" src="https://github.com/user-attachments/assets/0060a5c5-9873-4039-8489-1997952e9b81" />


</details>




# Citation
If you find our work valuable, we kindly ask you to consider citing our paper and starring ⭐ our repository. Our implementation includes dataset and useful tools and we hope it make life easier for the VOT research community.
```bibtex
@inproceedings{liu2024ntvot,
  title={NT-VOT211: A Large-Scale Benchmark for Night-time Visual Object Tracking},
  author={Yu Liu and Arif Mahmood and Muhammad Haris Khan},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  pages={to be announced},
  year={2024},
  organization={Springer}
}


@article{liu2026comprehensive,
  title={A Comprehensive Benchmark for Evaluating Night-time Visual Object Tracking},
  author={Liu, Yu and Mahmood, Arif and Khan, Muhammad Haris},
  journal={International Journal of Computer Vision},
  volume={134},
  pages={21},
  year={2026},
  publisher={Springer},
  doi={10.1007/s11263-025-02661-7},
  url={https://doi.org/10.1007/s11263-025-02661-7}
}

```
# Acknowledgments
The dataloader code borrows heavily from [PyTracking](https://github.com/visionml/pytracking).
# Maintenance
Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us.
# License
[MIT License](https://mit-license.org/)
