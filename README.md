# PIDS: Prior Image Guided Snapshot Spectral Compressive Imaging

- **Abstract**

*Spectral images with rich spatial and spectral information have wide usage, however, traditional spectral imaging techniques undeniably take a long time to capture scenes. We consider the computational imaging problem of the snapshot spectral spectrometer, i.e., the Coded Aperture Snapshot Spectral Imaging (CASSI) system. For the sake of a fast and generalized reconstruction algorithm, we propose a prior image guidance-based snapshot compressive imaging method. Typically, the prior image denotes the RGB measurement captured by the additional uncoded panchromatic camera of the dual-camera CASSI system. We argue that the RGB image as a prior image can provide valuable semantic information. More importantly, we design the Prior Image Semantic Similarity (PIDS) regularization term to enhance the reconstructed spectral image fidelity. In particular, the PIDS is formulated as the difference between the total variation of the prior image and the recovered spectral image. Then, we solve the PIDS regularized reconstruction problem by the Alternating Direction Method of Multipliers (ADMM) optimization algorithm. Comprehensive experiments on various datasets demonstrate the superior performance of our method.*

- **The formation process of the CASSI measurement and the uncoded RGB measurement.**
<div align=center><img width="500" height="246" src="https://github.com/YurongChen1998/Prior-Image-Guided-Snapshot-Spectral-Compressive-Imaging/blob/main/img/Fig1.jpg"/></div>

- **PIDS: Prior Image Semantic Similarity**

*PIDS regularization is a function* $: \mathbb{R}^{N} \times \mathbb{R}^{3HW} \rightarrow \mathbb{R}$ *that measures the difference between the L1-based total variation (TV) of two inputs* $x \in \mathbb{R}^{N}$ and $y^{PI} \in \mathbb{R}^{3HW}$,

$$PIDS(x, y^{PI}) \ \ = \ \ ||{\rm TV}(x) - {\rm TV}(y^{PI})||.$$

*We give the upper bound of PIDS as*

$$||{\rm TV}(x) - {\rm TV}(y^{PI})|| \ \ \leq \ \ ||{\rm TV}(x - y^{PI})||.$$

*The optimziation problem is formulated as* 

$$\hat{x} = \mathop{\arg\min}_{x} \frac{1}{2} ||y - Hx||_2^2 +  \lambda {\rm TV}(x - y^{PI}).$$

- **Implementation by Python （GPU accelerated version)**

```
# Solo utilizing PIDS regularization
from func import TV_minimization
x = TV_minimization(noisy_x, prior_img, tv_weight, tv_iter_max)
```


```
# Running spectral image reconstruction with PIDS regularization
python main_KAIST.py or python main_CAVE.py
```
