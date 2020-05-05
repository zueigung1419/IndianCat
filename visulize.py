import torch
from torch.distributions import Bernoulli, Beta
from torchvision.utils import make_grid, save_image


class Visualizer():
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.save_img = True

    def traverse_line(self, cont_dim=0, disc_dim=None, size=10, use_prior=True, set_zero=True, traverse=False, file_name=None):
        if set_zero:
            cont_samples = torch.zeros(size, self.args.cont).cuda()
        else:
            cont_samples = torch.randn(size, self.args.cont).cuda()
        fixed_value = torch.linspace(-2, 2, size).cuda()
        cont_samples[:, cont_dim] = fixed_value
        if use_prior:
            v_prior = Beta(torch.ones_like(cont_samples) * self.args.alpha_0, torch.ones_like(cont_samples))
            mask_prob = torch.cumprod(v_prior.sample(), dim=1)
            mask = Bernoulli(mask_prob).sample()
            cont_samples = cont_samples * mask
        if self.args.model_type != 'ibp':
            disc_samples = torch.zeros(size, self.args.disc).cuda()
            if traverse:
                for i in range(size):
                    disc_samples[i, i%self.args.disc] = 1.0
            else:
                disc_samples[:, disc_dim] = 1.0
            samples = torch.cat([cont_samples, disc_samples], dim=-1)
        else:
            samples = cont_samples
        with torch.no_grad():
            x = self.model.decoder(samples).view(-1, self.args.img_channel, self.args.img_size, self.args.img_size)
        if self.save_img:
            save_image(x.data, file_name, nrow=size, padding=0, pad_value=0.0)
        else:
            return make_grid(x.data, nrow=size, padding=0, pad_value=0.0)

    def traverse_grid(self, cont_dim=0, nrow=8, ncol=8, traverse=True, use_prior=True, set_zero=True, file_name=None):
        if traverse and self.args.disc != 0:
            nrow = self.args.disc
        if set_zero:
            cont_samples = torch.zeros(nrow*ncol, self.args.cont).cuda()
        else:
            cont_samples = torch.randn(nrow*ncol, self.args.cont).cuda()
        fixed_value = torch.linspace(-2, 2, ncol).cuda()
        for row in range(nrow):
            for i in range(ncol):
                cont_samples[i+row*ncol, cont_dim] = fixed_value[i]

        if use_prior:
            v_prior = Beta(torch.ones_like(cont_samples) * self.args.alpha_0, torch.ones_like(cont_samples))
            mask_prob = torch.cumprod(v_prior.sample(), dim=1)
            mask = Bernoulli(mask_prob).sample()
            cont_samples = cont_samples * mask

        if self.args.model_type != 'ibp':
            disc_samples = torch.zeros(nrow*ncol, self.args.disc).cuda()
            for i in range(nrow):
                for j in range(ncol):
                    disc_samples[j+i*ncol, i] = 1.0
            samples = torch.cat([cont_samples, disc_samples], dim=-1)
        else:
            samples = cont_samples
        with torch.no_grad():
            x = self.model.decoder(samples).view(-1, self.args.img_channel, self.args.img_size, self.args.img_size)
        if self.save_img:
            save_image(x.data, file_name, nrow=ncol, padding=0, pad_value=0.0)
        else:
            return make_grid(x.data, nrow=ncol, padding=0, pad_value=0.0)


