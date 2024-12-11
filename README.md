# NV-VOT211 ![Static Badge](https://img.shields.io/badge/The_Most_Challenging_VOT_Benchmark-ACCV2024%20Oral-blue)


The official implementation for the ACCV 2024 paper \[[_NT-VOT211: A Large-Scale Benchmark for Night-time Visual Object Tracking_](https://arxiv.org/abs/2410.20421)\]
![introc-1](https://github.com/user-attachments/assets/1a0e046a-0153-4c6f-acf2-5ba3e0c12b19)

Yu Liu,  [Arif Mahmood](https://scholar.google.com.sg/citations?user=_e6yGs4AAAAJ&hl=en),  [Muhammad Haris Khan](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en)
## Abstract
>Many current visual object tracking benchmarks such as OTB100, NfS, UAV123, LaSOT, and GOT-10K, predominantly contain day-time scenarios while the challenges posed by the night-time has been less investigated. It is primarily because of the lack of a large-scale, well-annotated night-time benchmark for rigorously evaluating tracking algorithms. To this end, this paper presents NT-VOT211, a new benchmark tailored for evaluating visual object tracking algorithms in the challenging night-time conditions.
NT-VOT211 consists of 211 diverse videos, offering 211,000 well-annotated frames with 8 attributes including camera motion, deformation, fast motion, motion blur, tiny target, distractors, occlusion and out-of-view.

## News(last update 2024-10-14) 
:fire::fire::fire: <sub>We will regularly update the links to the top three trackers right here</sub> :arrow_down:

| Tracker     | 	AUC | Tracker | Precision |
|:-----------:|:------------:|:-----------:|:-----------------:|
| [ProContEXT](https://github.com/jp-lan/procontext) | 40.10         | [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack)        | 55.80              |
| [KeepTrack](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#KeepTrack)| 39.60         | [KeepTrack](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#KeepTrack)      | 55.50              |
| [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) | 39.60         | [ProContEXT](https://github.com/jp-lan/procontext)        | 54.50              |

:fire::fire::fire: <sub>We will regularly update the links to the top three trackers right here</sub> :arrow_up:

We maintain two leaderboards: one is featured on [Papers with Code](https://paperswithcode.com/sota/video-object-tracking-on-nv-vot211),  and the other is on [EvalAI](https://eval.ai/web/challenges/challenge-page/2375/leaderboard/5892).  To submit a record on EvalAI, please adhere to the following instructions.

# How to benchmark:
## Download the dataset
You can download our dataset [here](https://zenodo.org/records/13768180?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNzA1ODYxMCwiZXhwIjozMjUwNTQwNzk5OX0.eyJpZCI6IjQwOWY4OGU3LWU3YjMtNDQ3OS1hMTAzLTg1ODBmZTI0MDkxNSIsImRhdGEiOnt9LCJyYW5kb20iOiI4NDVhMzgzNzEwZTQxZjEwZWE1ZmVhYWJkY2M4N2M4NyJ9.7LuMtijWPL-fCaTBbRpyXC0hS3R_UEljpgjkQBUIlf1ssU4JIFPXukuIlZejbdKGXqTZ3rMy9irIO7k85Ehzdw).
## Run on our dataset
Please follow these step-by-step [instructions](https://github.com/LiuYuML/NV-VOT211/tree/main/misc/dataloader) to evaluate your algorithm.
## Evaluation on server
**Challenge Phases:**
Please note that our challenge is divided into two phases: the **public phase** and the **private phase**.

- In the **private phase**, you are allowed to submit a maximum of **100 submissions per day**.
- In the **public phase**, you are limited to a maximum of **1 submission per month**.

We have implemented these limits to maintain a clean and readable leaderboard. We encourage all challengers to submit only their most competitive results on the public leaderboard.
To evaluate your results, please follow this [tutorial](https://github.com/LiuYuML/NV-VOT211/tree/main/misc/evaluation%20server).
# Annotation tool and meta information
Annotation tool and meta information can be found [here](https://github.com/LiuYuML/NV-VOT211/tree/main/misc/Other).

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
```
# Acknowledgments
The dataloader code borrows heavily from [PyTracking](https://github.com/visionml/pytracking).
# Maintenance
Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us.
# License
[MIT License](https://mit-license.org/)
