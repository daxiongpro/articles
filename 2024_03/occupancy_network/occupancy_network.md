# Occupancy Network

去年已经调研过关于 Occupancy Nerwork 的相关团队及其论文:[国内外自动驾驶研究团队收集资料](../../2023_05/domestic_autonomous_driving_research_team/domestic_autonomous_driving_research_team.md)。此文是关于 Occupancy Network 的一些总结。

## SurroundOcc

SurroundOcc: 取连续几帧点云得到稠密的点云，作为标注。

## TPVFormer

TPVFormer 是 CVPR2023 的一篇论文。其基于 CVPR2022 的 3D Unet。但是 3D Unet 慢，难以扩展到环视。

### 参考资料

* [TPVFormer——b 站](https://www.bilibili.com/video/BV1P54y1T7vS/?spm_id_from=333.337.search-card.all.click&vd_source=94ba2bc011dfd0f0eb6276bce9d70388)

## 3D Occupancy 挑战赛

CVPR 2023 有个 3D Occupancy 挑战赛，其 github 地址：[CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)。冠军算法是 [FB-BEV](https://github.com/NVlabs/FB-BEV)，其作者是 BEVFormer 一作李志琦(NVIDIA)，b 站有对 [FB-BEV 算法的讲解](https://www.bilibili.com/video/BV1PX4y1e7zz/?spm_id_from=333.337.search-card.all.click&vd_source=da7944bcc998e29818ec76ea9c6f1f47)。在挑战赛之后，开了个研讨会，在 b 站也有[全程录像](https://www.bilibili.com/video/BV1pN411D7au/?spm_id_from=333.337.search-card.all.click&vd_source=da7944bcc998e29818ec76ea9c6f1f47)。其先将了 3D Occupancy 发展历史，然后讲解了挑战赛榜单前三名的分享，最后有关于自动驾驶大模型的讨论。[知乎](https://zhuanlan.zhihu.com/p/638481909)有关于比赛方案总结。
