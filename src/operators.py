import torch
import torch.nn.functional as F
from functools import partial


def _nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)[0]
    output = torch.where(output == min_value, torch.nan, output)
    return output


def _nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)[0]
    output = torch.where(output == max_value, torch.nan, output)
    return output


def _nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def _nanstd(tensor, dim=None, keepdim=False):
    output = _nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def ops_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def ops_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def ops_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


def ops_inv(x: torch.Tensor) -> torch.Tensor:
    return 1 / x


def ops_rank(x: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(x, torch.nan)
    nan_mask = torch.isnan(x)
    valid_mask = ~nan_mask
    
    x_temp = x.clone()
    x_temp[nan_mask] = float('inf')
    
    ranks = torch.argsort(torch.argsort(x_temp, dim=1), dim=1) + 1
    ranks = ranks / valid_mask.sum(dim=1, keepdim=True)
    result[valid_mask] = ranks[valid_mask]
    return result


def ops_rolling_mean(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    return x_pad.unfold(dimension=0, size=window_size, step=1).nanmean(dim=2)


def ops_rolling_std(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(0, window_size, 1)
    return _nanstd(unfolded, dim=2)


def ops_rolling_max(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    return _nanmax(unfolded, dim=2)


def ops_rolling_min(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    return _nanmin(unfolded, dim=2)


def ops_pct_change(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_lag = ops_lag(x, window_size)
    result = torch.where(x_lag != 0, (x - x_lag) / x_lag, torch.nan)
    return result


def ops_lag(x: torch.Tensor, window_size: int) -> torch.Tensor:
    result = torch.full_like(x, torch.nan)
    if window_size < x.size(0):
        result[window_size:] = x[:-window_size]
    return result


def ops_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def ops_subtract(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def ops_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def ops_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(y != 0, x / y, torch.nan)


def ops_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(x, y)


def ops_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.min(x, y)


def ops_roll_corr(x: torch.Tensor, y: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    y_pad = F.pad(y, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    x_unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    y_unfolded = y_pad.unfold(dimension=0, size=window_size, step=1)
    
    x_mean = x_unfolded.nanmean(dim=2, keepdim=True)
    y_mean = y_unfolded.nanmean(dim=2, keepdim=True)
    
    cov = ((x_unfolded - x_mean) * (y_unfolded - y_mean)).nanmean(dim=2)
    x_std = _nanstd(x_unfolded, dim=2)
    y_std = _nanstd(y_unfolded, dim=2)
    
    corr = torch.where((x_std > 0) & (y_std > 0), cov / (x_std * y_std), torch.nan)
    return corr


def generate_operators(window_sizes):
    unary_ops = {
        'ops_abs': ops_abs,
        'ops_log': ops_log,
        'ops_neg': ops_neg,
        'ops_inv': ops_inv,
        'ops_rank': ops_rank
    }
    
    binary_ops = {
        'ops_add': ops_add,
        'ops_subtract': ops_subtract,
        'ops_multiply': ops_multiply, 
        'ops_divide': ops_divide,
        'ops_max': ops_max,
        'ops_min': ops_min
    }

    for window in window_sizes:
        for unary_name in ['ops_rolling_mean', 'ops_rolling_std', 'ops_rolling_max', 'ops_rolling_min', 'ops_pct_change', 'ops_lag']:
            unary_ops[f'{unary_name}_{window}'] = partial(eval(unary_name), window_size=window)
        for binary_name in ['ops_roll_corr']:
            binary_ops[f'{binary_name}_{window}'] = partial(eval(binary_name), window_size=window)

    return unary_ops, binary_ops