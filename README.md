# macroscopic-traffic-flow-model
This repository contains implementations of five common macroscopic traffic flow models for highway traffic simulation and prediction. Macroscopic traffic flow models treat traffic as a continuous fluid, simulating traffic dynamics by describing relationships between traffic density, speed, and flow.

# 宏观交通流模型库

## 项目概述

本仓库包含五种常用的宏观交通流模型实现，用于高速公路交通流仿真与预测。宏观交通流模型将交通流视为连续流体，通过描述交通密度、速度和流量的关系来模拟交通流动态特性。

## 模型介绍

### 1. METANET模型

METANET是一个基于二阶偏微分方程的宏观交通流模型，由Messmer和Papageorgiou于1990年提出。

**主要特点**：
- 二阶动态模型，同时描述密度和速度的演变
- 能够精确捕捉激波和拥堵形成过程
- 适用于复杂高速公路网络的交通状态预测
- 包含速度适应、密度变化和合并/分流等影响因素

### 2. LWR模型

LWR（Lighthill-Whitham-Richards）模型是最经典的宏观交通流模型，基于流体动力学原理。

**主要特点**：
- 基于一阶交通流保守方程
- 利用流量-密度基本图关系
- 计算简单高效
- 适合模拟基本交通流动态

### 3. CTM模型

CTM（Cell Transmission Model）是由Daganzo提出的LWR模型的离散实现版本。

**主要特点**：
- 将道路划分为一系列单元格
- 使用离散时间步长进行模拟
- 简单的发送-接收流量机制
- 易于实现和计算
- 支持匝道、交叉口等复杂路段建模

### 4. ML-CTM模型

ML-CTM（Machine Learning enhanced Cell Transmission Model）是结合机器学习技术的CTM模型增强版。

**主要特点**：
- 使用机器学习优化参数估计
- 自适应基本图关系
- 提高不确定条件下的预测准确性
- 支持多源数据融合

### 5. PI-LWR模型

PI-LWR（Predictive Incremental LWR）是专为短期交通流预测优化的LWR模型增强版。

**主要特点**：
- 采用Godunov数值求解方案
- 支持车型分类预测
- 考虑时间段特性（高峰期、平峰期）
- 支持自动参数校准
- 处理匝道影响
- 针对不同预测时间范围优化预测能力

## 安装与依赖

本仓库中的模型实现依赖以下Python库：

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
```


## 使用方法

### 数据格式

所有模型支持以下两种主要数据输入：

1. **路段数据**：包含道路网络拓扑结构的CSV文件，至少包含以下字段：
   - id: 路段ID
   - up_node: 上游节点ID
   - down_node: 下游节点ID
   - length: 路段长度(km)
   - lanes: 车道数
   - type: 路段类型

2. **交通流数据**：包含各门架点的交通流量信息，按照"trafficflow_{门架ID}.csv"格式命名，包含以下字段：
   - Time: 时间戳
   - B1-B3: 各类客车流量
   - T1-T3: 各类货车流量

### 运行示例

```bash
# 运行METANET模型
python metanet_model.py --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/metanet

# 运行LWR模型
python lwr_model.py --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/lwr

# 运行CTM模型
python ctm_model.py --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/ctm

# 运行ML-CTM模型
python ml_ctm_model.py --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/ml_ctm

# 运行PI-LWR模型（增强版1）
python pi_lwr_model.py --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/pi_lwr

# 运行PI-LWR模型（增强版2）
python pi_lwr_model.py --enhanced --road_data ./data/road.csv --flow_dir ./data/flow --output_dir ./results/pi_lwr_enhanced --calibrate
```

## 评估指标

所有模型使用以下指标评估预测性能：

- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均绝对百分比误差）

## 许可证

本项目采用MIT许可证。详情请参见LICENSE文件。

