import torch
import numpy as np

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """计算Top-1和Top-5准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def gps_to_meters(lat2, lon2, lat1, lon1):
    """特征工程：经纬度转米制位移"""
    dy = (lat2 - lat1) * 111320
    dx = (lon2 - lon1) * 111320 * np.cos(np.radians(lat1))
    return dx, dy