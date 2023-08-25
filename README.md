# PIDS: Prior Image Guided Snapshot Spectral Compressive Imaging

- **Abstract**

*Spectral images with rich spatial and spectral information have wide usage, however, traditional spectral imaging techniques undeniably take a long time to capture scenes. We consider the computational imaging problem of the snapshot spectral spectrometer, i.e., the Coded Aperture Snapshot Spectral Imaging (CASSI) system. For the sake of a fast and generalized reconstruction algorithm, we propose a prior image guidance-based snapshot compressive imaging method. Typically, the prior image denotes the RGB measurement captured by the additional uncoded panchromatic camera of the dual-camera CASSI system. We argue that the RGB image as a prior image can provide valuable semantic information. More importantly, we design the Prior Image Semantic Similarity (PIDS) regularization term to enhance the reconstructed spectral image fidelity. In particular, the PIDS is formulated as the difference between the total variation of the prior image and the recovered spectral image. Then, we solve the PIDS regularized reconstruction problem by the Alternating Direction Method of Multipliers (ADMM) optimization algorithm. Comprehensive experiments on various datasets demonstrate the superior performance of our method.*

---

- **The formation process of the CASSI measurement and the uncoded RGB measurement.**
<div align=center><img width="500" height="246" src="https://github.com/YurongChen1998/Prior-Image-Guided-Snapshot-Spectral-Compressive-Imaging/blob/main/img/Fig1.jpg"/></div>

---

- **PIDS: Prior Image Semantic Similarity**

*PIDS regularization is a function* $: \mathbb{R}^{N} \times \mathbb{R}^{3HW} \rightarrow \mathbb{R}$ *that measures the difference between the L1-based total variation (TV) of two inputs* $x \in \mathbb{R}^{N}$ and $y^{PI} \in \mathbb{R}^{3HW}$,

$$PIDS(x, y^{PI}) \ \ = \ \ ||{\rm TV}(x) - {\rm TV}(y^{PI})||.$$

*We give the upper bound of PIDS as*

$$||{\rm TV}(x) - {\rm TV}(y^{PI})|| \ \ \leq \ \ ||{\rm TV}(x - y^{PI})||.$$

*The optimization problem is formulated as* 

$$\hat{x} = \mathop{\arg\min}_{x} \frac{1}{2} ||y - Hx||_2^2 +  \lambda {\rm TV}(x - y^{PI}).$$

---

- **Data Preparation**

Download: [Google drive](https://drive.google.com/drive/folders/17LzFSdVCU2p0pfxbyghjZJw2nlf0yN97?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1mdLWXgvzkmQscfZu4t4M7A) (Code: u4er)

Put the Data into *Dataset* folder

---

- **Implementation by Python （GPU accelerated version)**

```
# Solo utilizing PIDS regularization
from func import TV_minimization

x = TV_minimization(noisy_x, prior_img, tv_weight, tv_iter_max)
```


```
# Running spectral image reconstruction with PIDS regularization
# One can adjust these hyperparameters for getting better results
#                          tv_weight in main_KAIST.py or main_CAVE.py
#                          tv_iter_max in main_KAIST.py or main_CAVE.py
#                          gamma in model.py
#                          alpha in func.py

python main_KAIST.py or python main_CAVE.py
```

```
In our work, we utilize the following settings:
    For KAIST S1:  tv_weight/tv_iter_max/alpha = 30/30/0.65
    For KAIST S2:  tv_weight/tv_iter_max/alpha = 45/30/0.55
    For KAIST S3:  tv_weight/tv_iter_max/alpha = 25/25/0.65
    For KAIST S4:  tv_weight/tv_iter_max/alpha = 25/25/0.50
    For KAIST S5:  tv_weight/tv_iter_max/alpha = 35/30/0.85
    For KAIST S6:  tv_weight/tv_iter_max/alpha = 35/30/0.95
    For KAIST S7:  tv_weight/tv_iter_max/alpha = 45/40/0.45
    For KAIST S8:  tv_weight/tv_iter_max/alpha = 27/30/0.45
    For KAIST S9:  tv_weight/tv_iter_max/alpha = 27/30/0.70
    For KAIST S10: tv_weight/tv_iter_max/alpha = 27/30/0.80

    For CAVE S1:   tv_weight/tv_iter_max/alpha = 25/25/0.95
    For CAVE S2:   tv_weight/tv_iter_max/alpha = 25/25/0.90
    For CAVE S3:   tv_weight/tv_iter_max/alpha = 02/14/0.80
    For CAVE S4:   tv_weight/tv_iter_max/alpha = 25/25/0.90
    For CAVE S5:   tv_weight/tv_iter_max/alpha = 25/25/0.70
    For CAVE S6:   tv_weight/tv_iter_max/alpha = 35/30/0.50
    For CAVE S7:   tv_weight/tv_iter_max/alpha = 35/30/0.85
    For CAVE S8:   tv_weight/tv_iter_max/alpha = 35/30/0.88
    For CAVE S9:   tv_weight/tv_iter_max/alpha = 30/30/0.88
    For CAVE S10:  tv_weight/tv_iter_max/alpha = 30/30/0.50
```


```
# Running spectral image reconstruction with TV regularization
# One can replace the code 'theta = TV_minimization(x1, ref_img, args.tv_weight, args.tv_iter_max)'
               by the code 'theta = TV_denoiser(x1, args.tv_weight, args.tv_iter_max)' in model.py
```

---
If you have any problem, please contact me: chenyurong1998 at outlook.com

If you use the code, please cite the paper:
```
@article{Yurong2023Prior,
  title={Prior Image Guided Snapshot Spectral Compressive Imaging},
  author={Yurong, Chen and Yaonan, Wang and Hui Zhang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi={10.1109/TPAMI.2023.3265749},
  year={2023},
  publisher={IEEE}
}
```

---

- **Acknowledge**

We greatly appreciate the help from Prof. He Wei (Wuhan University), Prof. Wang Lizhi (BIT), and Prof. Cao Xun (Nanjing University).
