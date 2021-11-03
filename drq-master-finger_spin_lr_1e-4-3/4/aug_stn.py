import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
        
class STN(nn.Module):
    def __init__(self, input_size, output_size=6, linear_size=32,
                 num_stage=2, p_dropout=0.5):
        super(STN, self).__init__()
        # print('point 0')
        self.linear_size = linear_size
        print('linear_size: {}'.format(linear_size))
        self.p_dropout = p_dropout
        print('p_dropout: {}'.format(p_dropout))
        self.num_stage = num_stage
        print('num_stage: {}'.format(num_stage))

        # noise dim
        self.input_size = input_size
        print('theta generator input dim: {}'.format(self.input_size))
        # theta dim
        self.output_size = output_size
        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # Initialize the weights/bias with identity transformation
        self.w2.weight.data.zero_()
        self.w2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.id_map = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float).cuda()
        self.id_map = torch.tensor([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], dtype=torch.float32).cuda()
        self.id_map_2 = torch.tensor([[[1, 0, 0],
                                    [0, 1, 0]]], dtype=torch.float32).cuda()
        self.pad_row = torch.tensor([[0., 0., 1.]], dtype=torch.float32).cuda()
        self.mse_loss = nn.MSELoss()


    def div_loss(self, theta):
        id_maps = self.id_map_2.repeat(theta.size(0), 1, 1)
        # print('id_maps shape: {}'.format(id_maps.size()))
        # print('theta shape: {}'.format(theta.size()))
        # exit()
        return self.mse_loss(theta, id_maps)

    def inv_theta(self, theta):
        pad_rows = self.pad_row.repeat(theta.size(0), 1, 1)
        theta_padded = torch.cat([theta, pad_rows], dim=1)
        theta_padded_inv = torch.inverse(theta_padded)
        theta_inv = theta_padded_inv[:, 0:-1, :]
        return theta_inv

    def tf_func(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x_tf = F.grid_sample(x, grid)
        return x_tf

    def diversity_loss(self, input1, output1, input2, output2, eps):
        output_diff = F.mse_loss(output1, output2, reduction='none')
        assert output_diff.size() == output1.size()
        output_diff_vec = output_diff.view(output_diff.size(0), -1).mean(dim=1)
        assert len(output_diff_vec.size()) == 1
        # noise_diff_vec = F.l1_loss(noise, noise_2, reduction='none').sum(dim=1)
        input_diff_vec = F.mse_loss(input1, input2, reduction='none').mean(dim=1)
        assert len(input_diff_vec.size()) == 1
        loss = output_diff_vec / (input_diff_vec + eps)
        # loss = torch.clamp(loss, max=1)
        return loss.mean()

    def theta_diversity_loss(self, noise, theta, eps=1e-3):
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        loss = self.diversity_loss(noise, theta, noise_2, theta_2, eps)
        return loss

    def img_diversity_loss(self, x, x_tf, noise, eps=1e-1):
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        x_tf_2 = self.tf_func(x, theta_2)
        # version 2
        loss = self.diversity_loss(noise, x_tf, noise_2, x_tf_2, eps)
        return loss

    def localization(self, noise):
        # pre-processing
        y = self.w1(noise)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        theta = self.w2(y)
        theta = theta.view(-1, 2, 3)

        return theta

    def forward(self, noise, x, label, require_loss=False):
        theta = self.localization(noise)
        assert theta.size(0) == x.size(0)
        theta_inv = self.inv_theta(theta)
        assert theta_inv.size() == theta.size()
        x_tf = self.tf_func(x, theta)
        x_tf_inv = self.tf_func(x, theta_inv)

        # get the transformed x and its corresponding label
        x_comb = torch.cat([x_tf, x_tf_inv], dim=0)
        label_comb = torch.cat([label, label], dim=0)

        if not require_loss:
            return x_comb, label_comb
        else:
            x_tf_recon = self.tf_func(x_tf, theta_inv)
            # reconstruct x from inverse theta tf
            x_tf_inv_recon = self.tf_func(x_tf_inv, theta)
            return x_comb, label_comb, \
                   self.mse_loss(x, x_tf_recon)+self.mse_loss(x, x_tf_inv_recon), \
                   self.img_diversity_loss(x, x_tf, noise)
# self.theta_cosine_diversity_loss(noise, theta)
# self.theta_diversity_loss(noise, theta)

##############