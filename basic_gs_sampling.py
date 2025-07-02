import numpy as np
import torch

def sampling(src_nodes, sample_num, neighbor_table):
    """
    根据源节点采样指定数量的邻居节点
    Args:
        src_nodes: 源节点列表
        sample_num: 需要采样的节点数
        neighbor_table: 邻接表，字典形式
    Returns:
        np.ndarray: 采样结果
    """
    results = []
    for sid in src_nodes:
        neighbors = neighbor_table.get(sid, [])
        if len(neighbors) == 0:
            res = np.array([sid] * sample_num)  # 如果无邻居，采样自身
        else:
            res = np.random.choice(neighbors, size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()

def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """
    多阶采样
    Args:
        src_nodes: 源节点列表
        sample_nums: 每层采样的邻居数量列表
        neighbor_table: 邻接表，字典形式
    Returns:
        list: 每层采样的节点列表
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result


def sampling_gpu(src_nodes: torch.Tensor, sample_num: int, neighbor_table: dict):
    """
    GPU优化的邻居采样函数
    Args:
        src_nodes: 源节点ID张量 (shape: [batch_size], device='cuda')
        sample_num: 需要采样的邻居数
        neighbor_table: 邻接表字典，值需为GPU张量
    Returns:
        torch.Tensor: 采样结果 (shape: [batch_size * sample_num], device='cuda')
    """
    batch_size = src_nodes.size(0)
    device = src_nodes.device

    # 预分配结果张量
    results = torch.empty(batch_size * sample_num, dtype=torch.long, device=device)
    
    # 并行化处理
    for i in range(batch_size):
        node_id = src_nodes[i].item()
        neighbors = neighbor_table.get(node_id, None)
        
        if neighbors is None or len(neighbors) == 0:
            # 无邻居时填充自身ID
            results[i*sample_num : (i+1)*sample_num] = node_id
        else:
            # 使用GPU加速的随机采样
            indices = torch.randint(0, len(neighbors), (sample_num,), device=device)
            results[i*sample_num : (i+1)*sample_num] = neighbors[indices]
    
    return results

def multihop_sampling_gpu(src_nodes: torch.Tensor, sample_nums: list, neighbor_table: dict):
    """
    GPU优化的多跳采样函数
    Args:
        src_nodes: 初始源节点ID张量 (shape: [batch_size], device='cuda')
        sample_nums: 每跳采样数列表 (如[10, 5]表示两跳采样)
        neighbor_table: 邻接表字典，值需为GPU张量
    Returns:
        List[torch.Tensor]: 各层采样结果列表，每个张量都在GPU上
    """
    sampling_result = [src_nodes]
    device = src_nodes.device

    # 将邻接表值转换为GPU张量（如果尚未转换）
    neighbor_table = {
        k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        for k, v in neighbor_table.items()
    }

    for hopk_num in sample_nums:
        # 执行当前跳采样
        current_src = sampling_result[-1]
        sampled = sampling_gpu(current_src, hopk_num, neighbor_table)
        
        # 展平并添加到结果列表
        sampling_result.append(sampled.view(-1))
    
    return sampling_result
