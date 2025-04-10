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

from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.functional import relu
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.data import Data, Batch
import torch.nn as nn



class IntraNodeAggConv(MessagePassing):
    r"""Based on the GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.

    :class:`GENConv` supports both :math:`\textrm{softmax}` (see
    :class:`~torch_geometric.nn.aggr.SoftmaxAggregation`) and
    :math:`\textrm{powermean}` (see
    :class:`~torch_geometric.nn.aggr.PowerMeanAggregation`) aggregation.
    Its message construction is given by:

    .. math::
        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_i +
        \mathrm{AGG} \left( \left\{
        \mathrm{ReLU} \left( \mathbf{x}_j + \mathbf{e_{ji}} \right) +\epsilon
        : j \in \mathcal{N}(i) \right\} \right)
        \right)

    .. note::

        For an example of using :obj:`GENConv`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            (:obj:`"softmax"`, :obj:`"powermean"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)
        t (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`1.0`)
        learn_t (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`t` for softmax aggregation dynamically.
            (default: :obj:`False`)
        p (float, optional): Initial power for power mean aggregation.
            (default: :obj:`1.0`)
        learn_p (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`p` for power mean aggregation dynamically.
            (default: :obj:`False`)
        msg_norm (bool, optional): If set to :obj:`True`, will use message
            normalization. (default: :obj:`False`)
        learn_msg_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor of message normalization. (default: :obj:`False`)
        norm (str, optional): Norm layer of MLP layers (:obj:`"batch"`,
            :obj:`"layer"`, :obj:`"instance"`) (default: :obj:`batch`)
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        expansion (int, optional): The expansion factor of hidden channels in
            MLP layers. (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_channels (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, Edge feature dimensionality is expected to match
            the `out_channels`. Other-wise, edge features are linearly
            transformed to match `out_channels` of node feature dimensionality.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'mean',
        output_func: Optional[str] = "mlp", # either activation function as string or "mlp" for additional mlp layer
        intranode_aggregation: str = "spda", # either "spda", "sum", "mean", "max", etc
        intranode_aggregation_kwargs: Optional[dict] = {},
        edge_message: Optional[bool] = False,
        internode_message: Optional[bool] = True,
        pass_message: bool = True,
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = True,
        norm: str = None,
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_channels: Optional[int] = None,
        node_dim: int = 0, # for proper propagation with Node tensor features
        flow = "source_to_target",
        internal_weights = True,
        **kwargs,
    ):

        if output_func != "mlp":
            raise NotImplementedError("Only MLP output function is supported for now as this version concatenates all node features, internode messages, and intranode messages")

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        # Override args of aggregator if `aggr_kwargs` is specified
        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t,
                                             semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        super().__init__(aggr=aggr, node_dim = node_dim, flow = flow, **kwargs)

        self.in_channels = in_channels # input dim per node-class i.e. F on input
        self.out_channels = out_channels # output dim per node-class i.e. D on output
        self.eps = eps # epsilon value for message construction

        # Count number of channels required for output mlp
        pre_out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # MLP to apply to the internode message M_{i,j,k} = Agg(MLP(H_{j,k},H_{i,j,k}))
        self.internode_message = internode_message
        if self.internode_message:
            self.internode_message_mlp = MLP([2 * out_channels, out_channels], norm=norm, bias=bias)
            pre_out_channels += out_channels

        # Intranode aggregation type
        self.intranode_aggregation = intranode_aggregation # whether to apply self attention within nodes


        if self.intranode_aggregation in ["sdpa", "sum", "mean", "max", "min"]:
            self.intranode_agg_func = IntraNodeAggregator(
                in_channels[0],
                out_channels,
                aggregation_type=intranode_aggregation,
                **(intranode_aggregation_kwargs or {})
            )
            pre_out_channels += out_channels

        else:
            self.intranode_agg_func = None


        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
        else:
            aggr_out_channels = out_channels

        if aggr_out_channels != out_channels or internal_weights:
            self.lin_aggr_out = Linear(aggr_out_channels, out_channels,
                                       bias=bias)


        channels = [int(pre_out_channels)] # 3 * out_channels because we concatenate the node features, internode message, and intranode message
        for i in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)

        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

        # if self.intranode_aggregation == "spda":
        #     self.intranode_agg_func = SDPA_layer(in_channels[0], out_channels)



        if edge_message: #also updating edge_attr
            self.edge_message = True
            self.edge_update_mlp = MLP([3 * out_channels, out_channels], norm=norm, bias=bias)
        else:
            self.edge_message = False
            self.edge_update_mlp = None



        if not isinstance(output_func, str):
            self.output_func = output_func()
        elif output_func == "mlp":
            self.output_func = MLP(channels, norm=norm, bias=bias)
        elif output_func == "relu":
            self.output_func = ReLU()
        else:
            raise NotImplementedError(f"Activation function {output_func} not implemented")



    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            self.lin_edge.reset_parameters()
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj = None,
                edge_attr: OptTensor = None, size: Size = None) -> Tuple[Tensor, Tensor]:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.intranode_agg_func is not None:
            intranode_message = self.intranode_agg_func(x[0])


        if self.internode_message:
            assert edge_index is not None
            # message passing + aggregation
            # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
            internode_message = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) # edge message e.g. AGG(MLP(H_{j,k},e_{i,j,k}))


            # If message normalization is needed
            if hasattr(self, 'msg_norm'):
                h = x[1] if x[1] is not None else x[0]
                assert h is not None
                internode_message = self.msg_norm(h, internode_message)


        out = x[0]

        if self.internode_message:
            out = cat([out, internode_message], dim=-1)

        if self.intranode_agg_func is not None:
            out = cat([out, intranode_message], dim=-1)

        node_output = self.output_func(out)



        if self.edge_message:
            edge_attr = self.edge_updater(edge_index, x = node_output, edge_attr = edge_attr)

        return node_output, edge_attr


    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        No modification needed here from the original GENConv implementation
        :param x_j:
        :param edge_attr:
        :return:
        """
        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        if edge_attr is not None:
            assert x_j.size(-1) == edge_attr.size(-1)
            if x_j.shape != edge_attr.shape: # means that edge attr is only a M, Fe tensor and not M, K, Fe
                edge_attr = edge_attr.unsqueeze(1).expand(x_j.shape)

        return self.internode_message_mlp(cat([x_j, edge_attr], dim=-1))




    def edge_update(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.edge_update_mlp(cat([x_i, x_j, edge_attr], dim=-1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0., output_func = ReLU):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(LazyBatchNorm1d(channels[i], affine=True))
                    # m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(output_func())
                m.append(Dropout(dropout))

        super().__init__(*m)


def initialize_weights(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         nn.init.zeros_(m.bias)
    # elif isinstance(m, nn.Conv2d):
        if isinstance(m, Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class FeedForwardNetwork(Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation = ReLU):
        super(FeedForwardNetwork, self).__init__()

        self.input_layer = Linear(input_dim, hidden_dim)
        self.hidden_layers = ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(num_layers-2)])
        self.output_layer = Linear(hidden_dim, output_dim)
        self.activation = activation()

        self.apply(initialize_weights)

    def forward(self, X):
        X = self.activation(self.input_layer(X))
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        return self.output_layer(X)

class SDPA_layer(Module):
    r"""
    Scaled Dot Product Attention Layer for GNN to be applied to any node embedding tensor X


    """

    def __init__(self, input_dim, output_dim):
        super(SDPA_layer, self).__init__()
        if input_dim != output_dim:
            self.proj = Linear(input_dim, output_dim)
        else:
            self.proj = None

        self.qkv_mlp = Linear(output_dim, output_dim * 3, bias=False)
        # self.mlp = Linear(output_dim, output_dim)
        self.mlp = FeedForwardNetwork(output_dim, output_dim, output_dim, 2, activation = ReLU)

        self.apply(initialize_weights)


    def forward(self, X):
        if self.proj is not None:
            X = self.proj(X)
        QKV = self.qkv_mlp(X)
        Q, K, V = QKV.chunk(3, dim=-1)
        H = sdpa(Q, K, V)
        out = self.mlp(H+X)
        return  out + X


class IntraNodeAggregator(Module):
    """
    Flexible intra-node aggregation layer that can apply different aggregation functions
    to node feature matrices, where each row of the output is an aggregation of all other rows.

    Supports multiple aggregation types:
    - "sdpa": Scaled dot product attention
    - "sum": Sum aggregation
    - "mean": Mean aggregation
    - "max": Max aggregation
    - "min": Min aggregation
    """

    def __init__(self, input_dim,
                 output_dim,
                 aggregation_type="sdpa",
                 mlp = True,
                 **kwargs):
        super(IntraNodeAggregator, self).__init__()

        self.aggregation_type = aggregation_type

        # Projection if input and output dimensions differ
        if input_dim != output_dim:
            self.proj = Linear(input_dim, output_dim)
        else:
            self.proj = None

        # SDPA-specific layers
        if aggregation_type == "sdpa":
            self.qkv_mlp = Linear(output_dim, output_dim * 3, bias=False)


        # MLP for post-processing (common for all aggregation types)
        if mlp:
            self.mlp = FeedForwardNetwork(output_dim, output_dim, output_dim, 2, activation=ReLU)

        self.apply(initialize_weights)

    def forward(self, X):
        # Project input if needed
        if self.proj is not None:
            X = self.proj(X)

        if hasattr(self, 'mlp'):
            X = self.mlp(X)

        # Apply the appropriate aggregation
        if self.aggregation_type == "sdpa":
            # Scaled dot product attention
            QKV = self.qkv_mlp(X)
            Q, K, V = QKV.chunk(3, dim=-1)
            H = sdpa(Q, K, V)

        elif self.aggregation_type == "sum":
            # For each row, sum all other rows
            N = X.size(1)  # Number of rows in the feature matrix
            # Create a mask to exclude self-aggregation
            mask = ~torch.eye(N, device=X.device).bool()
            # Expand mask to match batch dimensions
            if X.dim() > 2:
                mask = mask.unsqueeze(0).expand(X.size(0), -1, -1)
            # Apply mask and sum
            masked_X = X.unsqueeze(1).expand(-1, N, -1, -1) * mask.unsqueeze(-1)
            H = masked_X.sum(dim=2)

        elif self.aggregation_type == "mean":
            # For each row, average all other rows
            N = X.size(1)  # Number of rows in the feature matrix
            # Create a mask to exclude self-aggregation
            mask = ~torch.eye(N, device=X.device).bool()
            # Expand mask to match batch dimensions
            if X.dim() > 2:
                mask = mask.unsqueeze(0).expand(X.size(0), -1, -1)
            # Apply mask and mean
            masked_X = X.unsqueeze(1).expand(-1, N, -1, -1) * mask.unsqueeze(-1)
            H = masked_X.sum(dim=2) / (N - 1)

        elif self.aggregation_type == "max":
            # For each row, take max of all other rows
            N = X.size(1)  # Number of rows
            # Create expanded view with self set to minimum value
            expanded_X = X.unsqueeze(1).expand(-1, N, -1, -1)
            # Create mask for self-indices
            self_mask = torch.eye(N, device=X.device).bool()
            if X.dim() > 2:
                self_mask = self_mask.unsqueeze(0).expand(X.size(0), -1, -1)
            # Set diagonal elements to minimum value
            expanded_X = expanded_X.masked_fill(self_mask.unsqueeze(-1), float('-inf'))
            # Take max over the appropriate dimension
            H = expanded_X.max(dim=2)[0]

        elif self.aggregation_type == "min":
            # For each row, take min of all other rows
            N = X.size(1)  # Number of rows
            # Create expanded view with self set to maximum value
            expanded_X = X.unsqueeze(1).expand(-1, N, -1, -1)
            # Create mask for self-indices
            self_mask = torch.eye(N, device=X.device).bool()
            if X.dim() > 2:
                self_mask = self_mask.unsqueeze(0).expand(X.size(0), -1, -1)
            # Set diagonal elements to maximum value
            expanded_X = expanded_X.masked_fill(self_mask.unsqueeze(-1), float('inf'))
            # Take min over the appropriate dimension
            H = expanded_X.min(dim=2)[0]

        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation_type}")
        #
        # if self.residuals:
        #     H = H + X
        # if hasattr(self, 'mlp'):
        #     H = self.mlp(H)
        # if self.residuals:
        #     H = H + X
        return H
