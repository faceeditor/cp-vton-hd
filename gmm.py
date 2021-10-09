import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

# ----------------------------------------------------------------------------------------------------------------------
#                                                  GMM-related classes
# ----------------------------------------------------------------------------------------------------------------------
class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super(FeatureExtraction, self).__init__()

        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        layers += [nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)]
        layers += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        # Reshape features for matrix multiplication.

        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)

        # Perform matrix multiplication.

        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)

        return corr


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super(FeatureRegression, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU()
        )
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)


class TpsGridGen(nn.Module):
    def __init__(self, opt, dtype=torch.float):
        super(TpsGridGen, self).__init__()

        # Create a grid in numpy.
        # TODO: set an appropriate interval ([-1, 1] in CP-VTON, [-0.9, 0.9] in the current version of VITON-HD)
        grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, opt.fine_width), np.linspace(-0.9, 0.9, opt.fine_height))
        grid_X = torch.tensor(grid_X, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)
        grid_Y = torch.tensor(grid_Y, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)

        # Initialize the regular grid for control points P.
        self.N = opt.grid_size * opt.grid_size
        coords = np.linspace(-0.9, 0.9, opt.grid_size)
        # FIXME: why P_Y and P_X are swapped?
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base = P_X.clone()
        P_Y_base = P_Y.clone()

        Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)

        self.register_buffer('grid_X', grid_X, False)
        self.register_buffer('grid_Y', grid_Y, False)
        self.register_buffer('P_X_base', P_X_base, False)
        self.register_buffer('P_Y_base', P_Y_base, False)
        self.register_buffer('Li', Li, False)
        self.register_buffer('P_X', P_X, False)
        self.register_buffer('P_Y', P_Y, False)

    # TODO: refactor
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        return Li

    # TODO: refactor
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))

        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])

        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)

        return torch.cat((points_X_prime,points_Y_prime),3)

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        return warped_grid


class GMM(nn.Module):
    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()

        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_size=2 * opt.grid_size**2)
        self.gridGen = TpsGridGen(opt)

    def forward(self, inputA, inputB):
        featureA = F.normalize(self.extractionA(inputA), dim=1)
        featureB = F.normalize(self.extractionB(inputB), dim=1)

        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)

        warped_grid = self.gridGen(theta)
        return warped_grid
