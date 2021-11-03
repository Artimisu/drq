import torch
import torch.nn as nn
import math
import torch.nn.function as F
from aug_stn import STN
aug_net = STN(config.noise_dim, linear_size=config.linear_size)
aug_net = nn.DataParallel(aug_net)
aug_net_optim.zero_grad()
input, input_preaug, target
input_aug, target_aug, div_loss, diversity_loss = \
                model.aug_net(noise, input, target, require_loss=True)
aug_net_optim.step()



output_aug = model.target_net(input_aug, 'stn')
loss_div = div_loss * model.args.div_weight_stn
loss_diversity = -diversity_loss * model.args.diversity_weight_stn
loss_aug_net = loss_aug + loss_div + loss_diversity
loss_aug_net.backward()
model.aug_net_optim.step()
losses_adv.update((loss_aug * (-model.args.adv_weight_stn)).item(), input.size(0))
losses_div.update(loss_div.item(), input.size(0))
losses_diversity.update(loss_diversity.item(), input.size(0))
