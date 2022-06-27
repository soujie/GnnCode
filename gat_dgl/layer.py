import torch
import torch.nn as nn

class GatLayer(nn.Module):
    def __init__(self) -> None:
        '''
        使用dgl 实现gat layer , 主要参考dgl 中gat 源码的同态图部分
        在消息传播框架下, gat 可拆分为:
        1. 创建消息 , 这里特指计算gat 的score
            原始的gat 中 , score 的定义为 a^T(Wh_i || Wh_j), 但其等价于 a_l^T Wh_i + a_r^T Wh_j , 分别记为 el , er.
        2. 更新边的信息 e 和 a, e= e_l + e_r , a 为log-softmax 后的加权得分
        3. 更新节点的信息 , 使用 u_mul_e 进行节点上特征的更新
        4. 信息聚合 , 加权求和.

        在多头注意力机制下, 原本的实现方法应该是构造多个attention 层 , 再将其结果进行聚合(concat、mean等与输入顺序无关的聚合操作).
        但在pyg 和
        '''
        
        super().__init__()