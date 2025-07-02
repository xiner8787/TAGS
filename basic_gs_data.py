from collections import namedtuple,defaultdict
import numpy as np
import csv
import random
import torch
import os.path as osp
import pickle
from typing import Optional
import itertools

# 定义数据结构
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask','test_mask','pred_mask'])

class ArcData(object):
    def __init__(self, data_root="arc_data", rebuild=False):
        # self.data_root = data_root
        # save_file = osp.join(self.data_root, "processed_custom.pkl")
        
        # # 优先使用缓存数据（除非rebuild=True）
        # if osp.exists(save_file) and not rebuild:
        #     print("Using Cached file:", save_file)
        #     with open(save_file, "rb") as f:
        #         self._data = pickle.load(f)
        # else:
        #     self._data = self.process_data()
        #     with open(save_file, "wb") as f:
        #         pickle.dump(self._data, f)
        #     print("Cached file:", save_file)
        self._data = self.process_data()
        
    def process_data(self):
        """核心数据处理流程"""
        # 1. 加载原始数据
        node_features = self.load_node_features()
        node_labels = self.load_node_labels()  # 返回字典，空标签为None
        adjacency_dict = self.load_adjacency_dict()

        # 2. 构建特征矩阵
        num_nodes = len(node_features)
        x = np.array(list(node_features.values()), dtype=np.float32)  # 直接转换为numpy数组

        # 3. 构建标签向量（有效标签为int，空标签用-1标记）
        y = np.full(num_nodes, -1, dtype=np.int64)  # 初始化为-1
        valid_indices = []
        
        for idx, node_id in enumerate(node_features.keys()):
            label = node_labels.get(node_id)
            if label is not None:  # 只处理有效标签
                y[idx] = label
                valid_indices.append(idx)

        # 4. 划分掩码（仅使用有效标签索引）
        # 初始化所有mask为False
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        pred_mask = np.zeros(num_nodes, dtype=bool)

        # 设置随机种子
        self.set_seed(1)

        # 打乱有效索引
        np.random.shuffle(valid_indices)
        total_labeled = len(valid_indices)
        
        # 计算各集大小（按80-10-10比例）
        train_size = int(0.8 * total_labeled)
        val_size = int(0.1 * total_labeled)
        test_size = total_labeled - train_size - val_size

        # 分配掩码
        train_mask[valid_indices[:train_size]] = True
        val_mask[valid_indices[train_size:train_size+val_size]] = True
        test_mask[valid_indices[train_size+val_size:]] = True
        
        # 预测集：所有无效标签（y=-1）
        pred_mask = (y == -1)

        # 5. 验证数据有效性
        self._validate_masks(
            total_nodes=num_nodes,
            labeled_count=total_labeled,
            train_count=train_mask.sum(),
            val_count=val_mask.sum(),
            test_count=test_mask.sum(),
            pred_count=pred_mask.sum()
        )

        return Data(
            x=x, y=y, adjacency_dict=adjacency_dict,
            train_mask=train_mask, val_mask=val_mask,
            test_mask=test_mask, pred_mask=pred_mask
        )
    

    def load_node_labels(self, csv_file="2015_standardized_data.csv"):
        """加载节点标签，处理空值"""
        node_labels = {}
        
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                node_id = row['new_id']
                raw_label = row['Classification'].strip()  # 去除所有空白字符
                
                # 处理空标签（空字符串或空白字符）
                if not raw_label:
                    node_labels[node_id] = None
                else:
                    try:
                        node_labels[node_id] = int(raw_label)
                    except ValueError:
                        raise ValueError(
                            f"无效标签值: 文件{csv_file}中节点{node_id}的标签'{raw_label}'无法转换为整数"
                        f"\n请检查以下可能问题："
                            "\n1. 包含非数字字符（如字母、符号）"
                            "\n2. 数值超出整数范围"
                            "\n3. 存在隐藏空白字符（建议用文本编辑器检查）"
                        )
        return node_labels

    def _validate_masks(self, total_nodes, labeled_count, 
                       train_count, val_count, test_count, pred_count):
        """验证掩码划分的合理性"""
        # 基本完整性检查
        assert train_count + val_count + test_count == labeled_count, "有标签数据划分不完整"
        assert labeled_count + pred_count == total_nodes, "掩码总数与节点数不符"
        
        # 有效性检查
        if labeled_count == 0:
            raise RuntimeError("错误：数据集中没有有效标签！")
        if pred_count == 0:
            print("警告：数据集中所有节点都有标签，预测集为空")

        # 打印统计信息
        print("\n数据集统计:")
        print(f"总节点数: {total_nodes}")
        print(f"有标签节点: {labeled_count} ({labeled_count/total_nodes:.1%})")
        print(f"  训练集: {train_count} ({train_count/labeled_count:.1%})")
        print(f"  验证集: {val_count} ({val_count/labeled_count:.1%})")
        print(f"  测试集: {test_count} ({test_count/labeled_count:.1%})")
        print(f"预测集节点: {pred_count} ({pred_count/total_nodes:.1%})")

    def set_seed(self, seed):
        """设置全局随机种子"""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_node_features(self, csv_file="2015_standardized_data.csv"):
        """
        加载节点特征
        返回一个字典，键为建筑ID，值为特征向量
        max_rows 读取前多少行数据
        """
    # 初始化一个空字典data\10000normalized_features.csv
        node_features = {}

        # 打开 CSV 文件进行读取
        with open(csv_file, mode='r',encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)  # 读取文件，并跳过表头
            for row in csv_reader:
                # 从每一行中提取建筑 ID 和特征
                building_id = row['new_id']  # 假设列名为 'Id'
                excluded_columns = ['new_id', 'Id', 'ID_1', 'new_neighbors', 'neighbors','Classification']
                features = [float(row[key]) for key in row if key not in excluded_columns]
                node_features[building_id] = features  # 将建筑 ID 和特征列表加入字典
        
        return node_features



    def load_adjacency_dict(self,csv_file_path = "2015_standardized_data.csv"):
        """
        将 CSV 形式的邻居表转换为字典形式的邻接表。

        参数:
            csv_file_path (str): CSV 文件路径。

        返回:
            dict: 邻接表，键为节点 ID，值为邻居列表。
        """
        # adjacency_dict = defaultdict(list)  # 使用 defaultdict 初始化邻接表
        adjacency_dict = {}

        with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                building_id = int(row['new_id'])  # 当前节点 ID
                neighbors = row["new_neighbors"]  # 邻居列

                if neighbors:
                    neighbor_list = neighbors.split(";")  # 拆分为列表
                    for neighbor in neighbor_list:
                        if neighbor.strip():  # 忽略空值
                            neighbor_id = int(neighbor.strip())
                            if neighbor_id != building_id:  # 去掉自身的 ID
                                # 如果字典中没有该键，初始化一个空列表
                                if building_id not in adjacency_dict:
                                    adjacency_dict[building_id] = []
                                adjacency_dict[building_id].append(int(neighbor_id))  # 添加邻居

        return adjacency_dict
    
    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency_dict, train_mask, val_mask, test_mask, pred_mask"""
        return self._data

if __name__ == "__main__":
    # 实例化并打印信息
    data_loader = ArcData(rebuild=True)
    data = data_loader.data
    
    print("\n数据样例验证:")
    print("特征矩阵形状:", data.x.shape)
    print("标签示例:", data.y[:5])
    print("训练集首5节点:", data.train_mask[:5])
    print("预测集首5节点:", data.pred_mask[:5])
    print("邻接表样例:", dict(itertools.islice(data.adjacency_dict.items(), 3)))