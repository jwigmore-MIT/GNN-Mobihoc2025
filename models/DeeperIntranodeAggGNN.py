from typing import List, Optional, Tuple, Union
from torch_geometric.nn.conv import GENConv
from torch import Tensor
from torch import cat
from torch.nn import (
    BatchNorm1d,
    LazyBatchNorm1d,
    Dropout,
    InstanceNorm1d,
    LayerNorm,
    ReLU,
    Sigmoid,
    Sequential,
    Linear,
    Module,
    ModuleList,
)
import torch


import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.data import Data, Batch
from .IntraNodeAggConv import MLP, IntraNodeAggConv
import torch.nn as nn


class DeeperIntranodeAggGNN(Module):
    def __init__(self,
                 node_channels, # Node features
                 edge_channels, # Edge features
                 hidden_channels, # Hidden dimension
                 num_layers,      # Number of GNN Conv layers
                 output_channels = 1, # Number of outputs per edge
                 aggr = "mean",  # Aggregation Function
                 edge_message = False, # Whether to update edge attributes
                 conv_output_func = "mlp", # Output function of the convolution
                 output_activation = ReLU, # Activation function of edge decoder
                 edge_decoder = True, # Whether to use edge decoder
                 intranode_aggregation = "spda",# None, "spda", "sum" ... internode aggregation
                 intranode_aggregation_kwargs = {}, # kwargs for intranode aggregation
                 internal_weights = False # Whether to use internal weights in the convolution (otherwise should have mlp conv_output_func)
                 ):
        super().__init__()

        self.edge_message = edge_message
        self.node_encoder = Linear(node_channels, hidden_channels)
        self.edge_encoder = Linear(edge_channels, hidden_channels)

        self.intranode_aggregation = intranode_aggregation # whether to apply self attention within nodes

        self.layers = ModuleList()
        # Pass message = False for layer 0 as this just applies the edge encoder
        self.layers.append(IntraNodeAggConv(hidden_channels, hidden_channels,
                                            internode_message=False,
                                            intranode_aggregation = intranode_aggregation,
                                            intranode_aggregation_kwargs = intranode_aggregation_kwargs,))
        for i in range(1, num_layers+1):

            conv = IntraNodeAggConv(hidden_channels, hidden_channels,
                                     pass_message=True, edge_channels=hidden_channels,
                                     aggr = aggr, output_func = conv_output_func,
                                     edge_message = edge_message,
                                     intranode_aggregation = intranode_aggregation,
                                     intranode_aggregation_kwargs = intranode_aggregation_kwargs,
                                     internal_weights = internal_weights)


            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = ModDeepGCNLayer(conv, norm, act, block='res+', dropout=0.0,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
        if edge_decoder:
            self.edge_decoder = EdgeDecoder(hidden_channels, output_channels,
                                            activation = output_activation)
        else:
            self.lin = Linear(hidden_channels, output_channels)
        if output_channels == 1:
            self.scale_param = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale_param = None

        # self.apply(initialize_weights)

    def forward(self, data: Optional[Union[Data,Batch]] = None,
                x : Optional[Tensor] = None, edge_index: Optional[Tensor] = None, edge_attr: Optional[Tensor] = None):
        if data:
            if hasattr(data, 'data'): # to handle weird data handling of non tensor data by tensordict
                data = data.data
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            assert x is not None and edge_index is not None and edge_attr is not None
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.layers[0](x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x, edge_attr = layer(x, edge_index, edge_attr)

        x = self.layers[1].act(self.layers[1].norm(x))
        edge_attr = self.layers[1].act(self.layers[1].norm(edge_attr))

        if hasattr(self, 'edge_decoder'):
            z = self.edge_decoder(x, edge_index, edge_attr)
        else:
            z = self.lin(x)
        if not self.scale_param is None:
            # concatenate the scale parameter to the output
            # z would be shape (M, K, 1), scale_param is shape []
            z = torch.cat([z, self.scale_param.expand(z.shape)], dim = -1)
        return z



class ModDeepGCNLayer(torch.nn.Module):
    r"""The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

    * **Res+** (:obj:`"res+"`):

    .. math::
        \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
        \text{GraphConv}\to\text{Res}

    * **Res** (:obj:`"res"`) / **Dense** (:obj:`"dense"`) / **Plain**
      (:obj:`"plain"`):

    .. math::
        \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
        \text{Res/Dense/Plain}\to\text{Dropout}

    .. note::

        For an example of using :obj:`GENConv`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        conv (torch.nn.Module, optional): the GCN operator.
            (default: :obj:`None`)
        norm (torch.nn.Module): the normalization layer. (default: :obj:`None`)
        act (torch.nn.Module): the activation layer. (default: :obj:`None`)
        block (str, optional): The skip connection operation to use
            (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
            (default: :obj:`"res+"`)
        dropout (float, optional): Whether to apply or dropout.
            (default: :obj:`0.`)
        ckpt_grad (bool, optional): If set to :obj:`True`, will checkpoint this
            part of the model. Checkpointing works by trading compute for
            memory, since intermediate activations do not need to be kept in
            memory. Set this to :obj:`True` in case you encounter out-of-memory
            errors while going deep. (default: :obj:`False`)
    """
    def __init__(
        self,
        conv: Optional[Module] = None,
        norm: Optional[Module] = None,
        act: Optional[Module] = None,
        block: str = 'res+',
        dropout: float = 0.,
        ckpt_grad: bool = False,
    ):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """"""  # noqa: D419
        ### TODO: Fix this for edge messages as well
        args = list(args)
        x = args.pop(0)
        edge_index = args.pop(0)
        edge_attr = args.pop(0)


        if self.block == 'res+':
            h = x
            g = edge_attr
            if self.norm is not None:
                h = self.norm(h)
                g = self.norm(g)
            if self.act is not None:
                h = self.act(h)
                g = self.act(g)
            h = F.dropout(h, p=self.dropout, training=self.training)
            g = F.dropout(g, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h, g = checkpoint(self.conv, h, *(edge_index, g), use_reentrant=True,
                               **kwargs)
            else:
                h, g = self.conv(h, *(edge_index, g), **kwargs)

            return x + h, edge_attr + g

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h = checkpoint(self.conv, x, *args, use_reentrant=True,
                               **kwargs)
            else:
                h, edge_attr = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            return F.dropout(h, p=self.dropout, training=self.training), edge_attr

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'



class EdgeDecoder(MessagePassing):
    """
    Need to take X and edge_index and output edge_attr for each edge
    which is simply an MLP applied to the concatentation of the node features for each
    edge in edge_index
    """
    def __init__(self, input_dim, output_dim = 1, activation = ReLU):
        super(EdgeDecoder, self).__init__()
        self.mlp = MLP([3*input_dim, output_dim], norm=None, bias=True, dropout=0.0, output_func = activation)

    def forward(self, X, edge_index, edge_attr = None):
        edge_attr = cat([X[edge_index[0]], X[edge_index[1]], edge_attr], dim = -1)
        edge_attr = self.mlp(edge_attr)
        return edge_attr

def get_x_variation(x):
    """
    Look at the variation in the node embeddings x, where x[i] is the embedding of node i

    I want to look at
    (x[i] - x[j]).mean( for all i, j in the graph
    """

    return (x.unsqueeze(2) - x.unsqueeze(1)).pow(2).mean()
