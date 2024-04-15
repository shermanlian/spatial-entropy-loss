import numpy as np
import math
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as transforms


########################
def gaussian1d(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1, sigma1=1.5, sigma2=1.5):
    _1D_window1 = gaussian1d(window_size, sigma1).unsqueeze(1)
    _1D_window2 = gaussian1d(window_size, sigma2).unsqueeze(1)
    _2D_window = _1D_window1.mm(_1D_window2.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def spatial_entropy(p, reduction='mean', eps=1e-10):
     en = -torch.sum(p * torch.log(p + eps), dim=1)
     if reduction=='mean':
         return en.mean(dim=0).mean()
     else:
         return en.sum(dim=0).mean()

########
#In a 3x3 neighborhood,
#compute the average of the values of the pixels excluding the central one. 
#########
def neighbor(x, weight):
    c = x.shape[1] #channels
    kernel = torch.ones(c, 1, 3, 3)
    kernel[:, :, 1, 1] = 0
    # weight = create_window(3, c, 1.5, 3.2)
    # weight = torch.arange(1, 10).reshape(1, 1, 3, 3)
    if weight is not None:
        kernel = weight * kernel
    kernel = kernel / kernel.sum()
    kernel = kernel.to(x.device)

    return F.conv2d(x, kernel, padding=1, groups=c)
##################

################# kernel function ##########
#https://en.wikipedia.org/wiki/Kernel_(statistics)
def gaussian(x):
    return 1 / np.sqrt(2 * np.pi) * torch.exp(-0.5 * x**2)

#derivative of sigmoid
def sigmoid1(x):
    s = torch.sigmoid(x)
    return s * (1 - s)

def sigmoid2(x):
    return 2 / np.pi * 1 / (torch.exp(x) + torch.exp(-x))


################### KDE ##################

def rule_of_thumb_h(xi):
    n = xi.shape[-1] * xi.shape[-2]
    sig = torch.std(xi).item()
    return 1.06 * sig * n**(-0.2)

# choose a kernel function for calculating (x-xi)/h
def kernel_func(x, xi, h=0.5, ktype='sigmoid1'):
    ''' 
    Inputs:
        x: torch array -> [batch, 2, n, c], usually n=L*L
        xi: torch image -> [batch, 2, c, h, w]
        h: bandwith
    '''
    #broadcast
    x = x.unsqueeze(-1).unsqueeze(-1) # [batch, 2, n, c, 1, 1]
    xi = xi.unsqueeze(2) # [batch, 2, 1, c, h, w]
    v = (x - xi) / h

    if ktype == 'sigmoid1':
        return sigmoid1(v)
    elif ktype == 'sigmoid2':
        return sigmoid2(v)
    elif ktype == 'gaussian':
        return gaussian(v)
    else:
        print('No kernel specified!')


def kde_func(x, xi, h=0.5, win_size=1, win_type="mean"): # gaussian or mean
    ''' 
    Inputs:
        x: torch array -> [batch, 2, n, c], usually n=L*L
        xi: torch image pair -> [batch, 2, c, h, w]
        h: bandwith
    '''
    b, d, n, c = x.shape # batch, 2, L*L, channels
    N = xi.shape[-2] * xi.shape[-1]
    K = kernel_func(x, xi, h) # [batch, 2, n, c, h, w]
    K2 = K[:, 0] * K[:, 1]
    if win_size <= 0:
        return torch.sum(K2, dim=[-2, -1]) / (N * h * 0.5)
    padding_size = win_size // 2
    K2 = K2.view(b, -1, K2.shape[-2], K2.shape[-1])
    #K2 = F.pad(K2, (padding_size, padding_size, padding_size, padding_size), "reflect")

    #### window based p 
    #### K2: [batch, n, c, h, w]
    if win_type=="mean":
        # kernel1 = torch.ones(n*c, 1, win_size, win_size) / (0.5 * h * win_size**2)
        kernel1 = torch.ones(n*c, 1, 1, 1) / (0.5 * h * 1**2)
        kernel2 = torch.ones(n*c, 1, 3, 3) / (0.5 * h * 3**2)
        kernel3 = torch.ones(n*c, 1, 5, 5) / (0.5 * h * 5**2)
    elif win_type == "gaussian":
        kernel = create_window(win_size, n*c) / (h * 0.5)
    kernel1 = kernel1.to(x.device)
    kernel2 = kernel2.to(x.device)
    kernel3 = kernel3.to(x.device)
    p1 = F.conv2d(K2, kernel1, padding=0, groups=n*c)
    p2 = F.conv2d(K2, kernel2, padding=1, groups=n*c)
    p3 = F.conv2d(K2, kernel3, padding=2, groups=n*c)
    p = torch.cat([p1, p2, p3], dim=-1)

    return  p.view(b, n, -1, p.shape[-2], p.shape[-1])

###########################################


def prepare_elements(tensor, L=16, weight=None, gray=False):

    ######## contruct image and neighbor ########
    tensor = tensor.clamp(0, 1) * (L - 1)
    if gray:
        tensor = tensor.mean(dim=1, keepdim=True)
    x_neighbor = neighbor(tensor, weight=weight)
    xi = torch.stack([tensor, x_neighbor], dim=1)

    ######## contruct dictionary ########
    b, c, h, w = tensor.shape
    xs = torch.linspace(0, L-1, steps=L)
    x1, x2 = torch.meshgrid(xs, xs, indexing='xy')
    x = torch.stack([x1.flatten(), x2.flatten()], dim=0).unsqueeze(0).unsqueeze(-1)
    x = x.repeat(b, 1, 1, c).to(tensor.device)

    return x, xi

def kde(tensor, L=64, weight=None, h=0.5):
    x, xi = prepare_elements(tensor, L, weight=weight)
    # h = rule_of_thumb_h(xi) if auto_h else 0.5
    return kde_func(x, xi, h) # [b, n, c, h, w]

def cross_entropy(p_real, p_est, reduction='mean', eps=1e-10):
    '''
    p_real: [b, n, c, h, w]
    '''
    ce = -torch.sum(p_real * torch.log(p_est), dim=1).mean(dim=[-1, -2, -3])
    if reduction=='mean':
        return ce.mean()
    else:
        return ce.sum()

def relative_entropy(p_real, p_est, reduction='mean', eps=1e-10):
    re = torch.sum(p_real * (torch.log(p_real + eps) - torch.log(p_est + eps)), dim=1).mean(dim=[-1, -2, -3])
    if reduction=='mean':
        return re.mean()
    else:
        return re.sum()

def hellinger(p, q, reduction='mean'):
    sqrt_p = torch.sqrt(p)
    sqrt_q = torch.sqrt(q)
    h = 1/torch.sqrt(2) * torch.sqrt(torch.sum((sqrt_p - sqrt_q)**2, dim=1)).mean(dim=[-1, -2, -3])
    if reduction=='mean':
        return h.mean()
    else:
        return h.sum()
#############################################

def im2tensor(im, batch=1):
    tensor = torch.tensor(im / 255).permute(2, 0, 1).unsqueeze(0).float() # normlize to [0-1]
    tensor = tensor.repeat(batch, 1, 1, 1)
    return tensor

#L=64 the number of bins
def main(im1, im2, L=64, batch=1):
    tensor1 = im2tensor(im1, batch)
    tensor2 = im2tensor(im2, batch)
    p1 = kde(tensor1, L) # [b, n, c, h, w]
    p2 = kde(tensor2, L) # [b, n, c, h, w]

    print(f'2D entropy: {cross_entropy(p1, p2):.4f}, sum: {p1.sum(1).mean().round()}')

if __name__ == "__main__":
    # im_gt = cv2.imread('resized_color.png')
    # im_b1 = cv2.GaussianBlur(im_gt, (3, 3), 0)
    # im_b2 = cv2.GaussianBlur(im_gt, (5, 5), 0)
    # im_b3 = cv2.GaussianBlur(im_gt, (7, 7), 0)
    # main(im_gt, im_b1)
    # main(im_gt, im_b2)
    # main(im_gt, im_b3)
    image_size = (3, 3)
    pink_color = (255, 192, 203)  # RGB颜色，粉色

    image = Image.new('RGB', image_size, pink_color)

    transform = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为张量
    ])
    tensor_image = transform(image)
    # print(tensor_image)
    print(neighbor(tensor_image))
