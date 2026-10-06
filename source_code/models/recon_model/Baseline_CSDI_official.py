"""Official CSDI architecture adapted to battery-voltage imputation.

The diffusion model, side-information layout, random target strategy,
target-only epsilon loss, quadratic beta schedule, and reverse sampler follow
the reference implementation at https://github.com/ermongroup/CSDI (commit
7f24a436f08d98853a6b43d4f7f04e5a65ecdf27).

The only task-level adaptation is an extra normalized-current side-information
channel, preserving this repository's historical ``forward(V, I, mode)``
interface.  CSDI remains an imputation baseline: non-zero voltage observations
are conditioning values and are preserved, not denoised.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class _DiffusionEmbedding(nn.Module):
    """Diffusion-step embedding used by the official CSDI network."""

    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    @staticmethod
    def _build_embedding(num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (
            torch.arange(dim) / max(dim - 1, 1) * 4.0
        ).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, diffusion_step):
        embedding = self.embedding[diffusion_step]
        embedding = F.silu(self.projection1(embedding))
        return F.silu(self.projection2(embedding))


class _ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_dim, n_heads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_dim, channels)
        self.cond_projection = _conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = _conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = _conv1d_with_init(channels, 2 * channels, 1)

        time_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=64,
            activation="gelu",
        )
        self.time_layer = nn.TransformerEncoder(time_layer, num_layers=1)

    def _forward_time(self, values, base_shape):
        batch_size, channels, features, seq_len = base_shape
        if seq_len == 1:
            return values
        values = values.reshape(
            batch_size, channels, features, seq_len
        ).permute(0, 2, 1, 3)
        values = values.reshape(batch_size * features, channels, seq_len)
        transformer_input = values.permute(2, 0, 1)
        if transformer_input.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ):
                values = self.time_layer(transformer_input).permute(1, 2, 0)
        else:
            values = self.time_layer(transformer_input).permute(1, 2, 0)
        return values.reshape(
            batch_size, features, channels, seq_len
        ).permute(0, 2, 1, 3).reshape(batch_size, channels, features * seq_len)

    def forward(self, values, side_info, diffusion_embedding):
        base_shape = values.shape
        batch_size, channels, features, seq_len = base_shape
        values = values.reshape(batch_size, channels, features * seq_len)

        diffusion = self.diffusion_projection(diffusion_embedding).unsqueeze(-1)
        time_input = values + diffusion
        time_input_max = float(time_input.detach().abs().max().item())
        hidden = self._forward_time(time_input, base_shape)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(
                "CSDI time transformer became non-finite with "
                f"input_max_abs={time_input_max:.6g}."
            )

        # The official feature transformer is an identity for target_dim=1.
        hidden = self.mid_projection(hidden)
        side = side_info.reshape(batch_size, side_info.size(1), features * seq_len)
        hidden = hidden + self.cond_projection(side)

        gate, value = torch.chunk(hidden, 2, dim=1)
        hidden = torch.sigmoid(gate) * torch.tanh(value)
        residual, skip = torch.chunk(self.output_projection(hidden), 2, dim=1)

        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        values = values.reshape(base_shape)
        return (values + residual) / math.sqrt(2.0), skip


class _DiffusionNetwork(nn.Module):
    def __init__(
        self,
        side_dim,
        diffusion_steps,
        layers=4,
        channels=64,
        n_heads=8,
        diffusion_dim=128,
    ):
        super().__init__()
        self.channels = channels
        self.diffusion_embedding = _DiffusionEmbedding(
            diffusion_steps, embedding_dim=diffusion_dim
        )
        self.input_projection = _conv1d_with_init(2, channels, 1)
        self.output_projection1 = _conv1d_with_init(channels, channels, 1)
        self.output_projection2 = _conv1d_with_init(channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                _ResidualBlock(side_dim, channels, diffusion_dim, n_heads)
                for _ in range(layers)
            ]
        )

    def forward(self, values, side_info, diffusion_step):
        batch_size, input_dim, features, seq_len = values.shape
        hidden = values.reshape(batch_size, input_dim, features * seq_len)
        hidden = F.relu(self.input_projection(hidden))
        hidden = hidden.reshape(batch_size, self.channels, features, seq_len)

        diffusion = self.diffusion_embedding(diffusion_step)
        skips = []
        for layer_index, layer in enumerate(self.residual_layers):
            try:
                hidden, skip = layer(hidden, side_info, diffusion)
            except FloatingPointError as error:
                raise FloatingPointError(
                    f"CSDI residual_layer={layer_index}: {error}"
                ) from error
            skips.append(skip)

        hidden = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        hidden = hidden.reshape(batch_size, self.channels, features * seq_len)
        hidden = F.relu(self.output_projection1(hidden))
        hidden = self.output_projection2(hidden)
        return hidden.reshape(batch_size, features, seq_len)


class Model(nn.Module):
    """Conditional CSDI baseline for imputing one voltage feature."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.diffusion_steps = int(getattr(args, "diffusion_steps", 50))
        self.time_dim = int(getattr(args, "csdi_official_time_dim", 128))
        self.feature_dim = int(getattr(args, "csdi_official_feature_dim", 16))
        self.n_samples = int(getattr(args, "csdi_n_samples", 1))
        self.sample_clip = float(getattr(args, "csdi_sample_clip", 10.0))
        self.i_min = float(getattr(args, "i_min", 0.0))
        self.i_max = float(getattr(args, "i_max", 6.2))
        self.sampler_clip_events = 0
        self.sampler_clipped_values = 0
        self._sampler_clip_warned = False

        if self.n_samples < 1:
            raise ValueError("csdi_n_samples must be at least 1")
        if not math.isfinite(self.sample_clip) or self.sample_clip <= 1.0:
            raise ValueError("csdi_sample_clip must be finite and greater than 1")
        if self.i_max <= self.i_min:
            raise ValueError("i_max must be greater than i_min")

        self.feature_embedding = nn.Embedding(1, self.feature_dim)
        # official time embedding + feature embedding + cond mask + current
        side_dim = self.time_dim + self.feature_dim + 2
        self.diffmodel = _DiffusionNetwork(
            side_dim=side_dim,
            diffusion_steps=self.diffusion_steps,
            layers=int(getattr(args, "csdi_layers", 4)),
            channels=int(getattr(args, "csdi_channels", 64)),
            n_heads=int(getattr(args, "csdi_official_heads", 8)),
            diffusion_dim=int(getattr(args, "csdi_diffusion_dim", 128)),
        )
        self.recon = self.diffmodel
        self.pred = nn.Identity()

        betas = torch.linspace(
            1e-4 ** 0.5, 0.5 ** 0.5, self.diffusion_steps
        ) ** 2
        alpha_hat = 1.0 - betas
        alpha = torch.cumprod(alpha_hat, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_hat", alpha_hat)
        self.register_buffer("alpha", alpha)

    def _normalize_phy(self, voltage):
        voltage_01 = (voltage - self.args.v_min) / (
            self.args.v_max - self.args.v_min
        )
        return voltage_01 * 2.0 - 1.0

    def _denormalize_phy(self, voltage):
        voltage_01 = (voltage + 1.0) * 0.5
        return voltage_01 * (self.args.v_max - self.args.v_min) + self.args.v_min

    def _normalize_current(self, current):
        current_01 = (current - self.i_min) / (self.i_max - self.i_min)
        return current_01 * 2.0 - 1.0

    @staticmethod
    def _observed_mask(voltage):
        return ((voltage.abs() > 1e-6) & ~torch.isnan(voltage)).float()

    @staticmethod
    def _time_embedding(positions, dim):
        embedding = torch.zeros(
            positions.size(0), positions.size(1), dim, device=positions.device
        )
        position = positions.unsqueeze(2)
        divisor = 1.0 / torch.pow(
            10000.0,
            torch.arange(0, dim, 2, device=positions.device) / dim,
        )
        embedding[:, :, 0::2] = torch.sin(position * divisor)
        embedding[:, :, 1::2] = torch.cos(position * divisor)
        return embedding

    def _random_cond_mask(self, observed_mask):
        """Official CSDI random target-mask strategy."""
        scores = torch.rand_like(observed_mask) * observed_mask
        flat_scores = scores.reshape(scores.size(0), -1)
        flat_observed = observed_mask.reshape(observed_mask.size(0), -1)
        for index in range(observed_mask.size(0)):
            ratio = float(torch.rand((), device=observed_mask.device))
            num_observed = int(flat_observed[index].sum().item())
            num_masked = round(num_observed * ratio)
            if num_masked:
                target_indices = flat_scores[index].topk(num_masked).indices
                flat_scores[index, target_indices] = -1.0
        return (flat_scores > 0).reshape_as(observed_mask).float()

    def _side_info(self, cond_mask, current):
        batch_size, _, seq_len = cond_mask.shape
        positions = torch.arange(
            seq_len, device=cond_mask.device, dtype=torch.float32
        ).unsqueeze(0).expand(batch_size, -1)
        time_embedding = self._time_embedding(positions, self.time_dim)
        time_embedding = time_embedding.unsqueeze(2)

        feature_embedding = self.feature_embedding(
            torch.zeros(1, device=cond_mask.device, dtype=torch.long)
        )
        feature_embedding = feature_embedding.view(1, 1, 1, -1).expand(
            batch_size, seq_len, 1, -1
        )
        side = torch.cat([time_embedding, feature_embedding], dim=-1)
        side = side.permute(0, 3, 2, 1)
        return torch.cat([side, cond_mask.unsqueeze(1), current.unsqueeze(1)], dim=1)

    @staticmethod
    def _diffusion_input(noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1.0 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    def compute_loss(self, epsilon_prediction, epsilon, target_mask):
        residual = (epsilon - epsilon_prediction) * target_mask
        return residual.square().sum() / target_mask.sum().clamp_min(1.0)

    def _impute_once_ddim(
        self,
        observed_data,
        cond_mask,
        side_info,
        sampling_steps,
    ):
        sample = torch.randn_like(observed_data)
        timestep_schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=sampling_steps,
            device=observed_data.device,
        ).round().long()
        for schedule_index, step_tensor in enumerate(timestep_schedule):
            step = int(step_tensor.item())
            diffusion_step = torch.full(
                (observed_data.size(0),),
                step,
                device=observed_data.device,
                dtype=torch.long,
            )
            diffusion_input = self._diffusion_input(
                sample, observed_data, cond_mask
            )
            epsilon_prediction = self.diffmodel(
                diffusion_input, side_info, diffusion_step
            )
            if not torch.isfinite(epsilon_prediction).all():
                raise FloatingPointError(
                    "CSDI DDIM epsilon prediction became non-finite at "
                    f"diffusion_step={step}."
                )
            alpha = self.alpha[step]
            predicted_clean = (
                sample
                - torch.sqrt(1.0 - alpha) * epsilon_prediction
            ) / torch.sqrt(alpha)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)

            if schedule_index + 1 < timestep_schedule.numel():
                next_step = int(timestep_schedule[schedule_index + 1].item())
                next_alpha = self.alpha[next_step]
                sample = (
                    torch.sqrt(next_alpha) * predicted_clean
                    + torch.sqrt(1.0 - next_alpha) * epsilon_prediction
                )
            else:
                sample = predicted_clean
            if not torch.isfinite(sample).all():
                raise FloatingPointError(
                    f"CSDI DDIM sample became non-finite after diffusion_step={step}."
                )
        return sample

    def _impute_once(
        self,
        observed_data,
        cond_mask,
        side_info,
        sampling_steps,
    ):
        if sampling_steps < self.diffusion_steps:
            return self._impute_once_ddim(
                observed_data,
                cond_mask,
                side_info,
                sampling_steps,
            )
        sample = torch.randn_like(observed_data)
        for step in range(self.diffusion_steps - 1, -1, -1):
            diffusion_step = torch.full(
                (observed_data.size(0),),
                step,
                device=observed_data.device,
                dtype=torch.long,
            )
            diffusion_input = self._diffusion_input(
                sample, observed_data, cond_mask
            )
            epsilon_prediction = self.diffmodel(
                diffusion_input, side_info, diffusion_step
            )
            if not torch.isfinite(epsilon_prediction).all():
                finite_rows = torch.isfinite(epsilon_prediction).reshape(
                    epsilon_prediction.size(0), -1
                ).all(dim=1)
                bad_rows = torch.nonzero(
                    ~finite_rows,
                    as_tuple=False,
                ).flatten().tolist()
                bad_sample = sample[~finite_rows]
                finite_sample = bad_sample[torch.isfinite(bad_sample)]
                finite_observed = observed_data[torch.isfinite(observed_data)]
                finite_side = side_info[torch.isfinite(side_info)]
                sample_range = (
                    f"min={finite_sample.min().item():.6g}, "
                    f"max={finite_sample.max().item():.6g}, "
                    f"max_abs={finite_sample.abs().max().item():.6g}"
                    if finite_sample.numel()
                    else "no finite values"
                )
                observed_range = (
                    f"min={finite_observed.min().item():.6g}, "
                    f"max={finite_observed.max().item():.6g}"
                )
                side_range = (
                    f"min={finite_side.min().item():.6g}, "
                    f"max={finite_side.max().item():.6g}"
                )
                raise FloatingPointError(
                    f"CSDI epsilon prediction became non-finite at "
                    f"diffusion_step={step}, rows={bad_rows}; "
                    f"input sample {sample_range}; observed {observed_range}; "
                    f"side_info {side_range}."
                )
            coefficient_1 = 1.0 / torch.sqrt(self.alpha_hat[step])
            coefficient_2 = (1.0 - self.alpha_hat[step]) / torch.sqrt(
                1.0 - self.alpha[step]
            )
            sample = coefficient_1 * (
                sample - coefficient_2 * epsilon_prediction
            )
            if step > 0:
                sigma = torch.sqrt(
                    (1.0 - self.alpha[step - 1])
                    / (1.0 - self.alpha[step])
                    * self.betas[step]
                )
                sample = sample + sigma * torch.randn_like(sample)
            if not torch.isfinite(sample).all():
                bad_rows = torch.nonzero(
                    ~torch.isfinite(sample).reshape(sample.size(0), -1).all(dim=1),
                    as_tuple=False,
                ).flatten().tolist()
                raise FloatingPointError(
                    f"CSDI sample became non-finite at diffusion_step={step}, "
                    f"rows={bad_rows}."
                )
            outside_clip = sample.abs() > self.sample_clip
            if outside_clip.any():
                clipped_values = int(outside_clip.sum().item())
                maximum = float(sample.abs().max().item())
                self.sampler_clip_events += 1
                self.sampler_clipped_values += clipped_values
                sample = sample.clamp(-self.sample_clip, self.sample_clip)
                if not self._sampler_clip_warned:
                    warnings.warn(
                        "CSDI reverse sampler exceeded its numerical guard: "
                        f"clipped {clipped_values} values at diffusion_step={step} "
                        f"(max_abs={maximum:.6g}, bound={self.sample_clip:g}). "
                        "Further events are counted in sampler_clip_events and "
                        "sampler_clipped_values.",
                        RuntimeWarning,
                    )
                    self._sampler_clip_warned = True
        return sample

    def forward(self, V, I, mode="recon_only_train", steps=1):
        # ``steps`` is part of the shared reconstruction API and controls the
        # flow-matching solver used by BatteryDiT.  It must not shorten CSDI's
        # discrete diffusion chain: the formal CSDI baseline always evaluates
        # all configured reverse-diffusion steps.
        del steps
        sampling_steps = self.diffusion_steps
        if V.dim() == 2:
            V = V.unsqueeze(-1)
        if I.dim() == 2:
            I = I.unsqueeze(-1)

        observed_mask = self._observed_mask(V).permute(0, 2, 1)
        observed_data = self._normalize_phy(
            torch.nan_to_num(V, nan=0.0)
        ).permute(0, 2, 1)
        current = self._normalize_current(
            torch.nan_to_num(I, nan=0.0)
        ).permute(0, 2, 1)

        if mode == "recon_only_train":
            cond_mask = self._random_cond_mask(observed_mask)
            side_info = self._side_info(cond_mask, current)
            batch_size = observed_data.size(0)
            diffusion_step = torch.randint(
                0, self.diffusion_steps, (batch_size,), device=V.device
            )
            noise = torch.randn_like(observed_data)
            alpha = self.alpha[diffusion_step].view(batch_size, 1, 1)
            noisy_data = torch.sqrt(alpha) * observed_data + torch.sqrt(
                1.0 - alpha
            ) * noise
            diffusion_input = self._diffusion_input(
                noisy_data, observed_data, cond_mask
            )
            epsilon_prediction = self.diffmodel(
                diffusion_input, side_info, diffusion_step
            )
            target_mask = (observed_mask - cond_mask).clamp_min(0.0)
            return (
                epsilon_prediction.permute(0, 2, 1),
                noise.permute(0, 2, 1),
                target_mask.permute(0, 2, 1),
            )

        if mode == "recon_only_sampling":
            # CSDI is an imputer: fully observed rows are returned unchanged.
            # Running reverse diffusion for those rows cannot alter the output
            # and is prohibitively wasteful in the legacy benchmark pipeline.
            reconstruction = observed_data.clone()
            rows_with_missing = (observed_mask < 0.5).reshape(
                observed_mask.size(0), -1
            ).any(dim=1)
            if rows_with_missing.any():
                cond_mask = observed_mask[rows_with_missing]
                observed_subset = observed_data[rows_with_missing]
                current_subset = current[rows_with_missing]
                side_info = self._side_info(cond_mask, current_subset)
                samples = torch.stack(
                    [
                        self._impute_once(
                            observed_subset,
                            cond_mask,
                            side_info,
                            sampling_steps,
                        )
                        for _ in range(self.n_samples)
                    ],
                    dim=1,
                )
                point_estimate = samples.median(dim=1).values
                reconstruction[rows_with_missing] = (
                    cond_mask * observed_subset
                    + (1.0 - cond_mask) * point_estimate
                )
            return reconstruction.permute(0, 2, 1)

        raise ValueError(f"Unsupported mode: {mode}")
