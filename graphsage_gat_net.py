import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AttentionNeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True, aggr_method="attn"):
        super(AttentionNeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        
        # 线性变换权重
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        # 注意力机制参数
        self.attn_weight = nn.Parameter(torch.Tensor(output_dim * 2, 1))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.attn_weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, src_node_features, neighbor_feature):
        # 首先对源节点和邻居节点进行线性变换
        src_transformed = torch.matmul(src_node_features, self.weight)
        neighbor_transformed = torch.matmul(neighbor_feature, self.weight)
        
        if self.aggr_method == "attn":
            # 计算注意力分数
            src_expanded = src_transformed.unsqueeze(1).expand_as(neighbor_transformed)
            attn_input = torch.cat([src_expanded, neighbor_transformed], dim=-1)
            
            # 计算注意力得分
            attn_scores = torch.matmul(torch.tanh(attn_input), self.attn_weight)
            attn_scores = F.softmax(attn_scores, dim=1)
            
            # 加权求和
            attn_output = (neighbor_transformed * attn_scores).sum(dim=1)
            
            neighbor_hidden = attn_output
        elif self.aggr_method == "mean":
            neighbor_hidden = neighbor_transformed.mean(dim=1)
        elif self.aggr_method == "sum":
            neighbor_hidden = neighbor_transformed.sum(dim=1)
        elif self.aggr_method == "max":
            neighbor_hidden = neighbor_transformed.max(dim=1).values
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.aggr_method))
        
        if self.use_bias:
            neighbor_hidden += self.bias
        
        return neighbor_hidden

    
class MultiHeadAttnAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim_per_head = output_dim // heads
        self.scale = self.dim_per_head**-0.5
        
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.kv_proj = nn.Linear(input_dim, 2*output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, src, neighbors):
        # src: [B, D], neighbors: [B, K, D]
        B, K, _ = neighbors.shape
        q = self.q_proj(src).view(B, self.heads, self.dim_per_head)  # [B, H, D/H]
        k, v = self.kv_proj(neighbors).chunk(2, -1)  # [B, K, D] each
        k = k.view(B, K, self.heads, self.dim_per_head).transpose(1,2)  # [B, H, K, D/H]
        v = v.view(B, K, self.heads, self.dim_per_head).transpose(1,2)  # [B, H, K, D/H]
        
        attn = (q.unsqueeze(2) @ k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, -1)  # [B, D]
        return self.out_proj(out)

class AttnSageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 aggr_neighbor_method="multihead_attn",  # 新增选项
                 heads=4,  # 新增参数
                 activation=F.relu,
                 aggr_hidden_method="sum"):
        super(AttnSageGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        
        # 根据聚合方法选择不同的聚合器
        if aggr_neighbor_method == "multihead_attn":
            self.aggregator = MultiHeadAttnAggregator(
                input_dim, hidden_dim, heads=heads)
        else:
            self.aggregator = AttentionNeighborAggregator(
                input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
            
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(src_node_features, neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.aggr_hidden_method))
        
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden
        
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_mask=None):
        # x: [N, dim]
        attn_out, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), 
                               key_padding_mask=adjacency_mask)
        x = self.norm1(x + self.dropout(attn_out.squeeze(0)))
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class trans_AttnGraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list, num_classes,
                dropout=0.2, heads=4, use_transformer=True, aggr_method="multihead_attn"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.num_classes = num_classes
        self.use_transformer = use_transformer
        
        # 增强的GCN层
        self.gcn = nn.ModuleList()
        for i in range(len(hidden_dim)):
            in_dim = input_dim if i==0 else hidden_dim[i-1]
            self.gcn.append(AttnSageGCN(
                in_dim, hidden_dim[i],
                aggr_neighbor_method=aggr_method,  # 选择聚合器类型
                heads=heads))  # 传递头数
            self.gcn.append(nn.LayerNorm(hidden_dim[i]))
            self.gcn.append(nn.Dropout(dropout))
        
        # Transformer增强模块
        if use_transformer:
            self.transformer = GraphTransformerLayer(
                hidden_dim[-1], heads=heads)
            
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[-1]//2, num_classes))
        
    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[3*l]  # 每层包含GCN+Norm+Dropout
            for hop in range(self.num_layers - l):
                src = hidden[hop]
                neighbors = hidden[hop+1].view(
                    len(src), self.num_neighbors_list[hop], -1)
                h = gcn(src, neighbors)
                # 添加残差连接
                if l > 0 and h.shape == src.shape:
                    h = h + src  
                next_hidden.append(h)
            # 应用LayerNorm和Dropout
            next_hidden = [self.gcn[3*l+1](h) for h in next_hidden]
            next_hidden = [self.gcn[3*l+2](h) for h in next_hidden]
            hidden = next_hidden
        
        # Transformer增强
        if self.use_transformer:
            final_features = self.transformer(hidden[0])
        else:
            final_features = hidden[0]
        
        return self.classifier(final_features)