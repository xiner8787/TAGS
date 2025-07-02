# 城市建筑物功能推理模型TAGS（Trans_attnGraphSAGE）

本项目用于城市建筑物实体知识图谱构建中的功能推理模块，基于图神经网络（GraphSAGE）并结合注意力机制和Transformer层来实现城市建筑物的功能预测。

## 📁 项目目录结构
    ├── basic_gs_data.py # 数据处理脚本（原始数据转图结构数据）                                                                                                                                               
    ├── basic_gs_sampling.py # 建筑物邻居多阶采样函数                                                                                                                                                        
    ├── graphsage_gat_net.py # GraphSAGE + 注意力机制 + Transformer 网络模型结构定义                                                                                                                          
    ├── trans_gat_graphsage.ipynb # 主训练&预测代码（使用 2021 年数据训练 + 预测其余年份）                                                                                                                      
    ├── trans_gat_graphsage_finalmodel.pt # 训练好的模型参数（PyTorch）                                                                                                                                     
    ├── predictions/ # 模型预测结果数据                                                                                                                                                                          
    └── 2009–2021_standardized_data.csv # 标准化建筑物原始数据

---

## 🔗 数据下载

由于数据文件较大，请使用以下百度网盘链接下载：

- 链接: [https://pan.baidu.com/s/1jhLh6nz0W-3s9bGRvo_bYg?pwd=tags](https://pan.baidu.com/s/1jhLh6nz0W-3s9bGRvo_bYg?pwd=tags)  
- 提取码: `tags`

**数据文件包括：**
- `predictions/`：模型对各年份建筑物功能的预测结果
- `2009–2021_standardized_data.csv`：标准化后的建筑物图数据（包括特征和邻接信息，2021年数据还包含部分建筑物标签）

---

## 📌 项目功能

- 构建建筑物图结构（节点表示建筑物，边表示空间邻近/功能相似）
- 支持多阶邻居采样，增强图结构表示能力
- 使用基于 GraphSAGE 和 Transformer 的融合网络进行功能分类
- 可视化模型训练过程和输出预测结果

---

## 🚀 快速开始

### 1. 安装依赖（推荐使用 Python 3.8+）

```bash
pip install -r requirements.txt
```

### 2. 运行主程序
你可以在 Jupyter Notebook 中运行：
```bash
jupyter notebook trans_gat_graphsage.ipynb
```
⚠️ 运行前请根据年份调整：

修改 basic_gs_data.py 中的数据文件名

修改 trans_gat_graphsage.ipynb 中的输出文件名

## 📊 项目特点
本项目采用融合型图神经网络结构进行建筑物功能推理，模型特点如下
-GraphSAGE：多阶邻居采样，支持不同阶数邻居采样                                                                                                                                                                  
-注意力机制聚合：用于采样的邻居节点间权重分配与关系建模，实现局部特征聚合                                                                                                                                           
-Transformer层：建立远程依赖，捕捉全局特征


## 📄 项目背景
本项目是武汉大学资源与环境科学学院 2025 年大学生创新创业训练计划课题《城市建筑物实体知识图谱构建与特征推理方法》的模型实现部分。聚焦于建筑物功能的跨年迁移推理建模与预测。

---

## 📬 联系方式
如有问题或合作需求，请联系：
📧 邮箱：2668797512@qq.com


## 📜 许可证 License
本项目代码开源，遵循 MIT License 协议，可自由复制、修改和引用，敬请注明来源。
