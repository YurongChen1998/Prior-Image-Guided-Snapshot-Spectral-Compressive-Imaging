# PIDS: Prior Image Guided Snapshot Spectral Compressive Imaging

- **Abstract**

*Spectral images with rich spatial and spectral information have wide usage, however, traditional spectral imaging techniques undeniably take a long time to capture scenes. We consider the computational imaging problem of the snapshot spectral spectrometer, i.e., the Coded Aperture Snapshot Spectral Imaging (CASSI) system. For the sake of a fast and generalized reconstruction algorithm, we propose a prior image guidance-based snapshot compressive imaging method. Typically, the prior image denotes the RGB measurement captured by the additional uncoded panchromatic camera of the dual-camera CASSI system. We argue that the RGB image as a prior image can provide valuable semantic information. More importantly, we design the Prior Image Semantic Similarity (PIDS) regularization term to enhance the reconstructed spectral image fidelity. In particular, the PIDS is formulated as the difference between the total variation of the prior image and the recovered spectral image. Then, we solve the PIDS regularized reconstruction problem by the Alternating Direction Method of Multipliers (ADMM) optimization algorithm. Comprehensive experiments on various datasets demonstrate the superior performance of our method.*

- **The formation process of the CASSI measurement and the uncoded RGB measurement.**
<div align=center><img width="650" height="400" src="https://github.com/YurongChen1998/YurongChen1998.github.io/blob/gh-pages/img/Photo/Changsha%20City/DSC01015.JPG"/></div>
