import torch
from torch import nn
import torch.nn.functional as F
from models.dualmil_modules.map_modules import get_padded_mask_and_weight
# from torch_geometric.nn import GATv2Conv, GCNConv, DenseGCNConv
import dgl
from dgl.nn import GATConv

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            if cfg.SQUEEZE:
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], [k, 1], s, p, d))
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], [1, k], s, p, d))
            else:
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

    def forward(self, x, mask):

        padded_mask = mask
        for i, pred in enumerate(self.convs):
            tmp_shape = x.shape
            x = torch.reshape(x, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = torch.reshape(x, (tmp_shape[0], tmp_shape[1], x.shape[-3], x.shape[-2], x.shape[-1]))
            masked_weight = masked_weight.unsqueeze(1)
            x = x * masked_weight
        return x



class GCN(nn.Module):

    def __init__(self, cfg):
        super(GCN, self).__init__()

        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        heads = cfg.HEADS
        input_sizes = [input_size] * len(hidden_sizes)
        
        self.convs = nn.ModuleList() 
        for i, (h) in enumerate(heads):
            self.convs.append(GATConv(input_sizes[i], hidden_sizes[i], h))


    def get_edge_index(self, T):

        K = 5
        N = (K + 1) // 2

        T_idx = list()
        for i in range(T):
            for j in range(T):
                T_idx.append([i, j])
        T_idx = torch.tensor(T_idx, dtype=torch.float32).cuda()
        adj_matrix = torch.cdist(T_idx, T_idx, p=2) < N

        return adj_matrix


    def forward(self, x, masks):

        tmp_shape = x.shape
        batch_size = tmp_shape[0]
        num_sent = tmp_shape[1]
        T = tmp_shape[-1]
        x = x.reshape(batch_size*num_sent, -1, T*T)
        x = x.transpose(2, 1)

        single_mask = masks[0].flatten()
        node_features = x[:, single_mask.bool(), :]

        posid = torch.nonzero(single_mask)[:, 0]

        tmp_mask = self.get_edge_index(T)
        tmp_mask = tmp_mask[posid][:, posid] # (T, T) -> (len(posid), len(posid))
        edge_index = tmp_mask.nonzero().T

        single_graph = dgl.graph((edge_index[0], edge_index[1]))

        graphs = list()
        for i in range(batch_size*num_sent):
            graphs.append(single_graph.clone())
        graphs = dgl.batch(graphs)

        node_features = node_features.reshape(batch_size*num_sent*len(posid), -1)
        for i, pred in enumerate(self.convs):
            node_features = pred(graphs, node_features)
            node_features = node_features.reshape(batch_size*num_sent*len(posid), -1)
            node_features = F.elu(node_features)
        node_features = node_features.reshape(batch_size*num_sent, len(posid), -1)

        return_node_features = torch.zeros(batch_size*num_sent, T*T, node_features.shape[-1]).cuda()
        return_node_features[:, posid, :] = node_features
        return_node_features = return_node_features.transpose(2, 1)
        return_node_features = return_node_features.reshape(tmp_shape)

        return return_node_features
