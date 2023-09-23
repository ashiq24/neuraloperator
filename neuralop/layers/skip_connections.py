import torch
from torch import nn
from .resample import resample

def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating", output_scaling_factor=None, kernel_size=11,
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    skip_type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return getattr(nn, f"Conv{n_dim}d")(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )
    elif skip_type.lower() == "identity":
        return nn.Identity()
    elif skip_type.lower() == "conv":
        return ConvSkip(in_features, out_features,kernel_size=kernel_size,\
                    output_scaling_factor=output_scaling_factor,bias=bias,n_dim=n_dim)
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x

class ConvSkip(nn.Module):
    def __init__(self, in_features, out_features,kernel_size,\
                    output_scaling_factor,bias=True,n_dim=2,padding_mode='circular'):
        super().__init__()
        self.in_channels = in_features
        self.out_channels = out_features
        self.kernel_size = 2*(kernel_size//2) + 1
        if output_scaling_factor is None:
            output_scaling_factor = [1]*n_dim
        elif isinstance(output_scaling_factor, (float, int)):
            output_scaling_factor = [float(self.output_scaling_factor)] * n_dim

        self.output_scaling_factor = output_scaling_factor

        self.stride = 1 #[max(int(1/i),1) for i in output_scaling_factor]
        self.padding = self.kernel_size//2
        self.padding_mode = padding_mode

        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=self.kernel_size,\
                            padding=self.padding,stride=self.stride,padding_mode=self.padding_mode, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


