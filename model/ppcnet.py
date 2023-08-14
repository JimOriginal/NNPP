import torch
import sys
sys.path.append(r'/home/jim/ssddisk/Buffer/cnn_path_planning/path-planning-cnn')
from torch import nn
from dataset.map_sample import MapSample
from img_processing.gaussian_kernel import get_gaussian
from model.pe import GaussianRelativePE

class PPCNet(nn.Module):
    """ Path Planning CNN Network """    
    def __init__(self, n_layers=3, gaussian_blur_kernel=0):
        super(PPCNet, self).__init__()
        
        class _conv_block(nn.Module):
            def __init__(self, in_channels, out_channels, activation=None, norm_first=False, transpose=False, last_output_padding=0):
                super(_conv_block, self).__init__()
                self.activation = activation if activation is not None else nn.ReLU()
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                if transpose:               
                    self.bn3 = nn.BatchNorm2d(out_channels // 2)     
                    self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3)
                    self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3)
                    self.conv3 = nn.ConvTranspose2d(out_channels, out_channels // 2, 3, stride=2, padding=2, output_padding=last_output_padding)
                else:
                    self.bn3 = nn.BatchNorm2d(out_channels * 2)
                    self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
                    self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
                    self.conv3 = nn.Conv2d(out_channels, out_channels * 2, 3, stride=2)
                if norm_first:
                    self._fw = nn.Sequential(
                        self.conv1, self.bn1, self.activation, 
                        self.conv2, self.bn2, self.activation,
                        self.conv3, self.bn3, self.activation)
                else:
                    self._fw = nn.Sequential(
                        self.conv1, self.activation, self.bn1,
                        self.conv2, self.activation, self.bn2,
                        self.conv3, self.activation, self.bn3)
        
            def forward(self, x):
                return self._fw(x)
        
        self.gaussian_blur_kernel= gaussian_blur_kernel#for 正则化
        if gaussian_blur_kernel > 0:
            gaussian_kernel = torch.tensor(get_gaussian(gaussian_blur_kernel, sigma=0, normalized=True), dtype=torch.float32).view(1, 1, gaussian_blur_kernel, gaussian_blur_kernel)
            self.blur = nn.Conv2d(1, 1, gaussian_blur_kernel, padding=gaussian_blur_kernel // 2, bias=False)
            self.blur.weight.data = gaussian_kernel
        else:
            self.blur = None
        self.pe = GaussianRelativePE(100)
        self.sigm = nn.Sigmoid()
        n_channels = [64 * (2 ** i) for i in range(n_layers)]
        self.conv_down = nn.ModuleList([_conv_block(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
        self.conv_up = nn.ModuleList([_conv_block(2 * c, 2 * c, transpose=True, last_output_padding=1 if i == len(n_channels) - 1 else 0) for i, c in enumerate(n_channels[::-1])])
        self.bottleneck = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def pe_forward(self, x, start, goal):
        
        zeros = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        
        pe_start = self.pe(zeros, start)
        pe_goal = self.pe(zeros, goal)
        
        return torch.cat([x, pe_start, pe_goal], dim=1)

    def forward(self, x, start, goal):
        """Forward pass

        Args:
            x (Tensor): (N, C, H, W) batch of 2d maps.
            start (Tensor): (N, 2) start position.
            goal (Tensor)): (N, 2) goal position.

        Returns:
            Tensor: score map. Score of a pixel should be proportional to the probability of belonging to the shortest path.
        """
        if self.gaussian_blur_kernel > 0:
            with torch.no_grad():
                x = self.blur(x)
        skip_conn = []
        x = self.pe_forward(x, start, goal)
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
            if i < len(self.conv_down) - 1:
                skip_conn.append(x.detach())
        for i, conv in enumerate(self.conv_up):
            x = conv(x)
            if i < len(skip_conn):
                x = x + skip_conn[-1 - i]
        x = self.bottleneck(x)
        #x = self.pe_forward(x, start, goal)        
        x = self.conv_out(x)
        x = self.sigm(x)
        return x
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)







if __name__ == '__main__':
    model = PPCNet(3)
    path = '/home/jim/ssddisk/Buffer/cnn_path_planning/path-planning-cnn/dataset/map_dataset/validation/0a0e124d-1206-4682-9bfd-1d2209d6ed14.pt'
    sample = MapSample.load(path)
    import cv2
    map = sample.bgr_map()
    cv2.imshow('map', map)
    
    print(sample.map.size())
    out = model(sample.map.unsqueeze(0).unsqueeze(0), sample.start.unsqueeze(0).long(), sample.goal.unsqueeze(0).long())
    out.sum().backward()
    # rs=torch.reshape(out,[100,100])
    # # cv2.imshow('result',rs)
    # print(out.size())
    # cv2.waitKey(500)
    print(count_parameters(model))