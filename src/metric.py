import torch
from .operators import ops_rank, _nanstd


class ICMetric:
    def __init__(self, ret: torch.Tensor, method='spearman'):
        self.ret = ret
        self.method = method
        self.N = (~torch.isnan(ret)).sum()
        self.device = ret.device

    def _mean_ic(self, factor: torch.Tensor):
        if self.method == 'spearman':
            x_rank = ops_rank(factor)
            y_rank = ops_rank(self.ret)
        elif self.method == 'pearson':
            x_rank = factor
            y_rank = self.ret

        x_s = (x_rank - x_rank.nanmean(dim=1, keepdim=True)) / _nanstd(x_rank, dim=1, keepdim=True)
        y_s = (y_rank - y_rank.nanmean(dim=1, keepdim=True)) / _nanstd(y_rank, dim=1, keepdim=True)
        return (x_s * y_s).nanmean(dim=1).nanmean()

    def __call__(self, factor: torch.Tensor):
        nan_proportion = torch.isnan(factor).sum() / self.N
        if nan_proportion > 0.2:
            return torch.tensor(0.0, device=self.device)
        else:
            ic = self._mean_ic(factor)
            return torch.abs(ic) if not torch.isnan(ic) else torch.tensor(0.0, device=self.device)
