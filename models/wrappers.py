from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torchrl.modules import NormalParamExtractor
from torchrl.modules import TanhModule
import torch


class NormalWrapper(TensorDictModule):
    """
    Converts a GNN model that takes in X, edge_index, edge_attr and outputs logits to a model that outputs loc and scale

    """
    def __init__(self, module, scale_bias = 1.0):
        super(NormalWrapper, self).__init__(module = module, in_keys=["X", "edge_index", "edge_attr"], out_keys=["loc", "scale"])

        self.normal_params = NormalParamExtractor(scale_mapping=f"biased_softplus_{scale_bias}")

    def forward(self, td):
        td["logits"] = self.module(x = td["X"],edge_index = td["edge_index"],edge_attr =  td["edge_attr"])
        if td["logits"].shape[-1] != 2:
            raise ValueError("Model output must have shape (..., 2) for loc and scale parameters")
        td["loc"], td["scale"] = self.normal_params(td["logits"])
        return td


class ExpWrapper(torch.nn.Module):
    """
    Applies an exponential transformation to the output of a model
    """
    def __init__(self):
        super(ExpWrapper, self).__init__()



    def forward(self, t):
        return t.exp()
#