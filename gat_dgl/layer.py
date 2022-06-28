import dgl
import torch
import torch.nn as nn

class GatLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=.2,
                 residual = False,
                 activation = None,
                 bias = True
                 ) -> None:
        '''
        使用dgl 实现gat layer , 主要参考dgl 中gat 源码的同构图部分.(DGL 框架下同构图也有srcnode和dstnode, 但二者只有在二部图下才有区别, 并且同构图下二者指向同一地址)
        在消息传播框架下, gat 可拆分为:
        1. 创建消息 , 这里特指计算gat 的score
            原始的gat 中 , score 的定义为 e^T(Wh_i || Wh_j), 但其等价于 e_l^T Wh_i + e_r^T Wh_j , 分别记为 el , er.
        2. 更新边的信息 e 和 a, e= e_l + e_r , a 为log-softmax 后的加权得分
        3. 更新节点的信息 , 使用 u_mul_e 进行节点上特征的更新
        4. 信息聚合 , 加权求和.

        在多头注意力机制下, 原本的实现方法应该是构造多个attention 层 , 再将其结果进行聚合(concat、mean等与输入顺序无关的聚合操作).
        但在实际实现中, 都采取类通道概念, 使用线形层将原始特征映射为 n_nodes x (n_heads * n_feat) , 再将其reshape 为 n_nodes x n_heads x n_feats
        通过按通道进行加权求和 实现各head 上的信息聚合.
        '''
        
        super().__init__()
        self._num_heads = num_heads
        self._out_feats = out_feats
        
        # 计算 whi
        self.fc=nn.Linear(in_feats,out_feats*num_heads,bias=False)
        # 整图训练下, 使用元素点乘和广播机制实现 alpha 的计算
        self.attn_l = nn.parameter.Parameter(torch.FloatTensor(size=(1,num_heads,out_feats)))
        self.attn_r = nn.parameter.Parameter(torch.FloatTensor(size=(1,num_heads,out_feats)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.LeakyRelu = nn.LeakyReLU(negative_slope)
        
        if residual:
            self.res_fc = nn.Identity() #单位映射
        else:
            self.register_buffer('res_fc',None)
        
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(size=(num_heads*out_feats,)))    
        else:
            self.register_buffer('bias',None)
            
        self.reset_parameter()
        self.activation = activation
    
    def reset_parameter(self):
        gain = nn.init.calculate_gain('leaky_relu') # 获取torch 默认推荐的超参数
        nn.init.xavier_normal_(self.fc.weight,gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        
        if self.bias is not None:
            nn.init.constant_(self.bias,0)
    
    def forward(self,
                graph:dgl.graph,
                feat:torch.Tensor):
        with graph.local_scope():
            if (graph.in_degrees()==0).any():
                raise ValueError('zeros in degree will product invalid output')
            n_nodes = feat.shape[:-1]
            feat = self.feat_drop(feat)
            feat = self.fc(feat).view(*n_nodes,self._num_heads,self._out_feats) # calculate whi ,reshape to n_nodes x n_heads x out_dim
            
            
            #计算 e 中的 el 和er  , 其基于向量化计算, 并按照通道进行sum 聚合, alpha_l^T Wh_i 
            el = (feat * self.attn_l).sum(-1).unsqueeze(-1) 
            er = (feat * self.attn_r).sum(-1).unsqueeze(-1)
            
            # 更新节点信息和边信息, 并计算 e = el+ er
            graph.ndata.update({'ft':feat,'el':el,'er':er}) # NOTE: 官方实现的 'ft':{{feat}} , 此处不对, 得修正一下
            graph.apply_edges(dgl.function.u_add_v('el','er','e'))
            e =self.LeakyRelu(graph.edata.pop('e')) # leak_relu(e^T(whi||whj))
            graph.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(graph,e))
            
            # 加权求和
            graph.update_all(dgl.function.u_mul_e('ft','a','m'),
                             dgl.function.sum('m','ft'))
            
            rst = graph.dstdata['ft']
            
            
            if self.res_fc is not None:
                res = self.res_fc(feat).view(*n_nodes,-1,self._out_feats)
                rst = rst + res 
            
            if self.bias is not None:
                rst = rst + self.bias.view(*((1,) * len(n_nodes)), self._num_heads, self._out_feats)
            
            if self.activation  :
                rst = self.activation(rst)
            
            return rst 
            
     
if __name__=='__main__':

    g = dgl.data.CoraFullDataset()[0]
    feat = g.ndata['feat']
    n_feats = feat.shape[-1]
    
    layer = GatLayer(n_feats,5,3)
    ans = layer(g,feat)
    print(ans.shape)
                  
            
            