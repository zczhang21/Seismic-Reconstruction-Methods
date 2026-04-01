# Seismic Data Interpolation

本项目汇总两种地震数据插值方法的实现：

- **MWNI** - Minimum Weighted Norm Interpolation
- **Radon** - Frequency Domain Anti-aliasing Greedy Radon Transform

## 方法说明

### MWNI

基于文献 [1] 实现，引用了 Sacchi 的代码框架。

- 在频率域通过最小加权范数求解，从不规则采样或缺失道数据中恢复完整波场
- 采用共轭梯度（CG）迭代求解
- 适用于一般缺失道插值场景

### Radon

参考文章 [2] 复现。

- 频率域抗混叠贪婪 Radon 变换
- 采用匹配追踪思想，每次迭代选取能量最强的 Radon 系数
- 适合处理线性或双曲同相轴的插值问题

## 参考文献

[1] Minimum weighted norm interpolation of seismic records

[2] Seismic data interpolation by frequency domain anti-aliasing greedy Radon transform
