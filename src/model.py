import torch
import torch.nn as nn
import pytorch_lightning as pl


class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_filters=16,
                 depth=3,
                 kernel_size=3,
                 padding=1,
                 learning_rate=1e-3,
                 use_smoothing=True,
                 use_tv_loss=True,
                 physics_loss_weight=0.0,
                 dx=0.002,
                 dy=0.002,
                 fluid_threshold=1e-6,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['mean', 'std'])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.padding = padding
        self.learning_rate = learning_rate
        self.use_smoothing = use_smoothing
        self.use_tv_loss = use_tv_loss
        self.physics_loss_weight = physics_loss_weight
        self.dx = dx
        self.dy = dy
        self.fluid_threshold = fluid_threshold

        # Store mean and std as buffers for device management
        # self.register_buffer('mean', mean)
        # self.register_buffer('std', std)

        # Build encoder blocks
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            out_channels_enc = base_filters * (2 ** i)
            self.encoders.append(self._convolution_block(prev_channels, out_channels_enc))
            prev_channels = out_channels_enc

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._convolution_block(base_filters * (2 ** (depth - 1)), base_filters * (2 ** depth))

        # Build decoder blocks
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            in_channels_dec = base_filters * (2 ** (i + 1))
            out_channels_dec = base_filters * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(in_channels_dec, out_channels_dec,
                                                   kernel_size=2, stride=2))
            self.decoders.append(self._convolution_block(in_channels_dec, out_channels_dec))

        # Final convolution
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        self.smoothing = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=5,
            padding=2,
            groups=out_channels,
            bias=False,
            padding_mode='replicate'
        )
        self.init_gaussian_kernel(self.smoothing, 5, 1.0)

        # Loss function
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def init_gaussian_kernel(layer, k, sigma):
        ax = torch.arange(k) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)
        with torch.no_grad():
            layer.weight.copy_(kernel.repeat(layer.out_channels, 1, 1, 1))

    def _convolution_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size,
                      padding=self.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size,
                      padding=self.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def tv_loss(pred):
        # Maybe penalize only for pressure channel (assumed to be channel index 3)
        # p = pred[:, 3:4, :, :]
        p = pred
        dx = p[:, :, 1:, :] - p[:, :, :-1, :]
        dy = p[:, :, :, 1:] - p[:, :, :, :-1]
        return dx.abs().mean() + dy.abs().mean()

    def loss_fn(self, pred, target, tv_weight=0.1):
        mse = self.mse_loss(pred, target)
        if self.use_tv_loss:
            tv = tv_weight * self.tv_loss(pred)
            return mse + tv
        else:
            return mse

    @staticmethod
    def bilinear_sample(field, coords):
        """Evaluate a field at continuous grid coordinates."""
        batch_size, channels, height, width = field.shape
        x = coords[..., 0]
        y = coords[..., 1]

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1
        wx = x - x0.to(x.dtype)
        wy = y - y0.to(y.dtype)

        flat = field.reshape(batch_size, channels, height * width)

        def gather(ix, iy):
            index = (iy * width + ix).unsqueeze(1).expand(-1, channels, -1)
            return torch.gather(flat, 2, index)

        f00 = gather(x0, y0)
        f10 = gather(x1, y0)
        f01 = gather(x0, y1)
        f11 = gather(x1, y1)

        wx = wx.unsqueeze(1)
        wy = wy.unsqueeze(1)
        return (
            (1 - wx) * (1 - wy) * f00
            + wx * (1 - wy) * f10
            + (1 - wx) * wy * f01
            + wx * wy * f11
        )

    @staticmethod
    def make_query_coordinates(pred):
        """Create one differentiable query point at each grid-cell center."""
        batch_size, _, height, width = pred.shape
        x = torch.arange(width - 1, device=pred.device, dtype=pred.dtype) + 0.5
        y = torch.arange(height - 1, device=pred.device, dtype=pred.dtype) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(1, -1, 2)
        return coords.expand(batch_size, -1, -1).clone().requires_grad_(True)

    def denormalize_for_physics(self, pred):
        """Restore physical output units without leaving the autograd graph."""
        normalizer = self.trainer.datamodule.normalizer
        if normalizer is None:
            return pred

        mean = torch.as_tensor(normalizer.mean, device=pred.device, dtype=pred.dtype)
        std = torch.as_tensor(normalizer.std, device=pred.device, dtype=pred.dtype)
        return pred * (std + normalizer.eps) + mean

    def fluid_query_mask(self, topology):
        """Keep query points whose four surrounding topology pixels are fluid."""
        lam = topology[:, 0]
        fluid = (
            (lam[:, :-1, :-1] < self.fluid_threshold)
            & (lam[:, :-1, 1:] < self.fluid_threshold)
            & (lam[:, 1:, :-1] < self.fluid_threshold)
            & (lam[:, 1:, 1:] < self.fluid_threshold)
        )
        return fluid.reshape(fluid.shape[0], -1)

    def divergence_loss(self, pred_norm, topology):
        """Mean squared in-plane divergence over pure-fluid grid cells."""
        pred = self.denormalize_for_physics(pred_norm)
        coords = self.make_query_coordinates(pred)
        values = self.bilinear_sample(pred, coords)

        grad_ux = torch.autograd.grad(
            values[:, 0].sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_uy = torch.autograd.grad(
            values[:, 1].sum(), coords, create_graph=True
        )[0]

        div_u = grad_ux[..., 0] / self.dx + grad_uy[..., 1] / self.dy
        fluid_mask = self.fluid_query_mask(topology).to(div_u.dtype)
        return (fluid_mask * div_u.square()).sum() / (fluid_mask.sum() + 1e-8)

    @staticmethod
    def denormalize_output(sample, mean, std, eps=1e-6):
        return sample * (std.view(-1, 1, 1) + eps) + mean.view(-1, 1, 1)

    @staticmethod
    def mask_lambda(x, y, lambda_index=0, mean=None, std=None):
        if mean is None:
            mean = torch.zeros(x.size(1), device=x.device)
        if std is None:
            std = torch.ones(x.size(1), device=x.device)

        # Boolean mask where lambda == 1
        mask = y[:, lambda_index, :, :] == 1

        # Compute the normalized value that corresponds to 0 in original scale
        zero_norm = -mean.view(-1, 1, 1) / std.view(-1, 1, 1)

        # Apply to first 3 channels
        x[:, :3, :, :][mask.unsqueeze(1).expand_as(x[:, :3, :, :])] = zero_norm[:3, :, :].unsqueeze(0)

        return x

    def forward(self, x, debug=False, denormalize=False, mean=None, std=None):
        enc_features = []
        out = x
        for i, encoder in enumerate(self.encoders):
            out = encoder(out)  # For weird dimensions, use out = pad_to_even(encoder(out))
            enc_features.append(out)
            if debug:
                print(f"e{i + 1}: {out.shape}")
            out = self.pool(out)

        out = self.bottleneck(out)
        if debug:
            print(f"bottleneck: {out.shape}")

        for i in range(self.depth):
            upconv = self.upconvs[i]
            decoder = self.decoders[i]
            up = upconv(out)
            skip = enc_features[self.depth - 1 - i]
            # for weird dimensions, use up = center_crop(up, skip)
            if debug:
                print(f"up{i + 1}: {up.shape}, skip{i + 1}: {skip.shape}")
            out = decoder(torch.cat([up, skip], dim=1))

        out = self.final(out)

        if self.use_smoothing:
            out = self.smoothing(out)

        if denormalize and (mean is not None) and (std is not None):
            out = self.denormalize_output(out, mean, std)

        # out = self.mask_lambda(out, inputs)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        data_loss = self.loss_fn(y_hat, y)
        div_loss = self.divergence_loss(y_hat, x)
        loss = data_loss + self.physics_loss_weight * div_loss

        self.log('training_data_loss', data_loss, on_step=False, on_epoch=True)
        self.log('training_div_loss', div_loss, on_step=False, on_epoch=True)
        self.log('training_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning. Computes the loss and logs it.
        """
        x, y = batch
        # Get model prediction
        y_hat = self(x)
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        # Log loss
        self.log('validation_loss', loss)

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_loss(self, x, y):
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss
