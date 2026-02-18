import torch
from torch.cuda.amp import custom_bwd, custom_fwd

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
def custom_mse_loss(predictions, targets, iteration=1):
    # if iteration == 208:
    #     breakpoint()
    # 计算预测值和真实值之间的差异
    differences = predictions - targets
    # 对差异进行平方运算
    squared_differences = differences ** 2
    
    # 计算平方差异的平均值
    mse = squared_differences.sum((1, 2, 3)).mean()
    return mse