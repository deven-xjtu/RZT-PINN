
import argparse
import hashlib
import os
from pathlib import Path
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.special import roots_legendre
from torch.quasirandom import SobolEngine
from tqdm import trange

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LX, LY, LOAD = 0.4, 0.2, 5e6
CORE_THICKNESS, FACE_THICKNESS = 0.02, 0.005
TOTAL_THICKNESS = 2 * FACE_THICKNESS + CORE_THICKNESS
OUTPUT_SCALE = 2e-4
ENERGY_NORMALIZER = 100.0
Z_LAYERS = np.array([
    -TOTAL_THICKNESS / 2,
    -CORE_THICKNESS / 2,
    CORE_THICKNESS / 2,
    TOTAL_THICKNESS / 2,
])
LAYER_THICKNESSES = np.diff(Z_LAYERS)

REDUCED_STIFFNESS_GPA = np.array([
    [213.42593320235756, 213.42593320235756, 66.02593320235754,
     73.7, 73.7, 73.7],
    [26.052599999999998, 26.052599999999998, 14.229935999999999,
     7.288175999999999, 18.64695310374319, 18.64695310374319],
    [213.42593320235756, 213.42593320235756, 66.02593320235754,
     73.7, 73.7, 73.7],
])
REDUCED_STIFFNESS = REDUCED_STIFFNESS_GPA * 1e9
G_XZ = REDUCED_STIFFNESS[:, 4]
G_YZ = REDUCED_STIFFNESS[:, 5]
G_AVG_X = TOTAL_THICKNESS / np.sum(LAYER_THICKNESSES / G_XZ)
G_AVG_Y = TOTAL_THICKNESS / np.sum(LAYER_THICKNESSES / G_YZ)
BETA_X = G_AVG_X / G_XZ - 1.0
BETA_Y = G_AVG_Y / G_YZ - 1.0

def validate_stiffness():
    core = REDUCED_STIFFNESS_GPA[1]
    if not (core[0] == core[1] and core[4] == core[5]):
        raise ValueError('The fixed core must satisfy Q11=Q22 and G13=G23.')
    for layer in REDUCED_STIFFNESS:
        if not (layer[0] > 0 and layer[1] > 0
                and layer[0] * layer[1] > layer[2] ** 2
                and np.all(layer[3:] > 0)):
            raise ValueError('The embedded reduced stiffness is not positive definite.')
    if not (np.isclose(BETA_X @ LAYER_THICKNESSES, 0.0, atol=1e-14)
            and np.isclose(BETA_Y @ LAYER_THICKNESSES, 0.0, atol=1e-14)):
        raise ValueError('The embedded stiffness produces an invalid RZT zigzag function.')


class FourierNet(nn.Module):
    def __init__(self, fourier_dim, fourier_scale, in_plane_power, w_power):
        super().__init__()
        self.in_plane_power = in_plane_power
        self.w_power = w_power
        if fourier_dim == 128 and np.isclose(fourier_scale, 0.2):
            center_generator = torch.Generator().manual_seed(42)
            frequencies = torch.randn(
                2, fourier_dim // 2, generator=center_generator, dtype=torch.float64
            ) * fourier_scale
        else:
            frequencies = torch.randn(2, fourier_dim // 2) * fourier_scale
        self.register_buffer('B', frequencies)
        self.fc1 = nn.Linear(fourier_dim, 128)
        self.res = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.fc_out = nn.Linear(128, 7)
        self.register_buffer('weights', torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]))

    def forward(self, points):
        angles = points @ self.B * (2 * np.pi)
        features = torch.cat((angles.sin(), angles.cos()), dim=1)
        hidden = torch.tanh(self.fc1(features))
        raw = self.fc_out(hidden + self.res(hidden)) * self.weights
        dist = ((1 - points[:, 0:1] ** 2) * (1 - points[:, 1:2] ** 2)).clamp_min(0)
        return torch.cat((
            raw[:, :2] * dist ** self.in_plane_power,
            raw[:, 2:3] * dist ** self.w_power,
            raw[:, 3:] * dist ** self.in_plane_power,
        ), dim=1)

    def fixed_inputs(self, points):
        with torch.no_grad():
            angles = points @ self.B * (2 * np.pi)
            sine, cosine = angles.sin(), angles.cos()
            features = torch.cat((sine, cosine), dim=1)
            tangents = tuple(
                torch.cat((cosine * self.B[j], -sine * self.B[j]), dim=1) * (2 * np.pi)
                for j in range(2)
            )
            raw_dist = (1 - points[:, 0:1] ** 2) * (1 - points[:, 1:2] ** 2)
            dist = raw_dist.clamp_min(0)
            distance_tangents = (
                -2 * points[:, 0:1] * (1 - points[:, 1:2] ** 2),
                -2 * points[:, 1:2] * (1 - points[:, 0:1] ** 2),
            )
            powers = dist.new_tensor(
                [self.in_plane_power] * 2 + [self.w_power] + [self.in_plane_power] * 4
            ).unsqueeze(0)
            factors = dist ** powers
            factor_slopes = powers * dist ** (powers - 1)
            factor_tangents = tuple(
                factor_slopes * tangent * (raw_dist >= 0)
                for tangent in distance_tangents
            )
        return features, tangents, factors, factor_tangents

    @staticmethod
    def linear(value, tangents, layer):
        derivative = torch.cat(tangents, dim=0) @ layer.weight.T
        return layer(value), derivative.chunk(2, dim=0)

    @staticmethod
    def tanh(value, tangents):
        value = torch.tanh(value)
        return value, tuple(tangent * (1 - value.square()) for tangent in tangents)

    def values_and_derivatives(self, inputs):
        features, tangents, factors, factor_tangents = inputs
        value, tangents = self.linear(features, tangents, self.fc1)
        value, tangents = self.tanh(value, tangents)
        skip, skip_tangents = value, tangents
        for layer in (self.res[0], self.res[2]):
            value, tangents = self.linear(value, tangents, layer)
            value, tangents = self.tanh(value, tangents)
        value = value + skip
        tangents = tuple(a + b for a, b in zip(tangents, skip_tangents))
        raw, tangents = self.linear(value, tangents, self.fc_out)
        raw = raw * self.weights
        tangents = tuple(tangent * self.weights for tangent in tangents)
        values = raw * factors
        derivatives = tuple(
            tangent * factors + raw * factor_tangent
            for tangent, factor_tangent in zip(tangents, factor_tangents)
        )
        return values, derivatives


class RZTPINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Lx, self.Ly, self.q0 = LX, LY, LOAD
        self.scale = OUTPUT_SCALE
        self.energy_normalizer = ENERGY_NORMALIZER
        self.adaptive_fraction = args.adaptive_fraction
        self.adaptive_centers_count = args.adaptive_centers
        self.adaptive_noise = args.adaptive_noise
        self.register_buffer('props', torch.tensor(REDUCED_STIFFNESS))
        self.register_buffer('z_layers', torch.tensor(Z_LAYERS))
        self.register_buffer('h_k', torch.tensor(LAYER_THICKNESSES))
        self.register_buffer('beta_x', torch.tensor(BETA_X))
        self.register_buffer('beta_y', torch.tensor(BETA_Y))
        self.net = FourierNet(
            args.fourier_dim,
            args.fourier_scale,
            args.in_plane_power,
            args.pinn_w_power,
        )
        self.sobol = SobolEngine(dimension=2, scramble=True, seed=args.seed)
        self.adaptive_points = None
        self.setup_integrators(args.gauss_order)

    def setup_integrators(self, gauss_order):
        thickness_points = torch.tensor([-0.7745966692414834, 0.0, 0.7745966692414834])
        thickness_weights = torch.tensor([5 / 9, 8 / 9, 5 / 9])
        all_z, all_weights, all_layers = [], [], []
        for layer in range(len(LAYER_THICKNESSES)):
            center = (self.z_layers[layer] + self.z_layers[layer + 1]) / 2
            half_height = self.h_k[layer] / 2
            all_z.append(center + thickness_points * half_height)
            all_weights.append(thickness_weights * half_height)
            all_layers.append(torch.full_like(thickness_points, layer, dtype=torch.long))
        self.register_buffer('int_z', torch.cat(all_z))
        self.register_buffer('int_w', torch.cat(all_weights))
        self.register_buffer('int_k', torch.cat(all_layers))

        points, weights = roots_legendre(gauss_order)
        point_tensor = torch.tensor(points)
        gx, gy = torch.meshgrid(point_tensor, point_tensor, indexing='ij')
        self.register_buffer('gauss_pts', torch.stack((gx.flatten(), gy.flatten()), dim=1))
        weight_tensor = torch.tensor(weights)
        self.register_buffer('gauss_wts', (weight_tensor[:, None] * weight_tensor[None, :]).flatten())

    def get_phi(self, z_values):
        phi_x = torch.zeros_like(z_values)
        phi_y = torch.zeros_like(z_values)
        for layer in range(len(self.h_k)):
            mask = ((z_values >= self.z_layers[layer] - 1e-9)
                    & (z_values <= self.z_layers[layer + 1] + 1e-9))
            if mask.any():
                offset_x = torch.sum(self.beta_x[:layer] * self.h_k[:layer]) if layer else 0.0
                offset_y = torch.sum(self.beta_y[:layer] * self.h_k[:layer]) if layer else 0.0
                phi_x[mask] = offset_x + self.beta_x[layer] * (
                    z_values[mask] - self.z_layers[layer]
                )
                phi_y[mask] = offset_y + self.beta_y[layer] * (
                    z_values[mask] - self.z_layers[layer]
                )
        return phi_x, phi_y

    def prepare_stiffness(self):
        with torch.no_grad():
            phi_x, phi_y = self.get_phi(self.int_z)
            stiffness = self.props.new_zeros((14, 14))
            indices = torch.tensor([0, 1, 3, 4, 5], device=self.props.device)
            for index in range(len(self.int_z)):
                layer, z_value = self.int_k[index], self.int_z[index]
                strain_map = self.props.new_zeros((5, 14))
                strain_map[0, 0], strain_map[0, 3], strain_map[0, 6] = 1, z_value, phi_x[index]
                strain_map[1, 1], strain_map[1, 4], strain_map[1, 7] = 1, z_value, phi_y[index]
                strain_map[2, 2], strain_map[2, 5] = 1, z_value
                strain_map[2, 8], strain_map[2, 9] = phi_x[index], phi_y[index]
                strain_map[3, 10], strain_map[3, 12] = 1, self.beta_x[layer]
                strain_map[4, 11], strain_map[4, 13] = 1, self.beta_y[layer]
                layer_stiffness = self.props[layer]
                constitutive = torch.diag(layer_stiffness[indices])
                constitutive[0, 1] = constitutive[1, 0] = layer_stiffness[2]
                stiffness += strain_map.T @ constitutive @ strain_map * self.int_w[index]
            self.register_buffer('thickness_stiffness', stiffness, persistent=False)
        self.cached_gauss_inputs = self.net.fixed_inputs(self.gauss_pts)

    def compute_energy(self, points, is_gauss=False, return_density=False):
        inputs = self.cached_gauss_inputs if is_gauss else self.net.fixed_inputs(points)
        with torch.set_grad_enabled(not return_density):
            output, (dx, dy) = self.net.values_and_derivatives(inputs)
            output = output * self.scale
            dx = dx * (self.scale * 2 / self.Lx)
            dy = dy * (self.scale * 2 / self.Ly)
            strains = torch.stack((
                dx[:, 0], dy[:, 1], dy[:, 0] + dx[:, 1],
                dx[:, 3], dy[:, 4], dy[:, 3] + dx[:, 4],
                dx[:, 5], dy[:, 6], dy[:, 5], dx[:, 6],
                dx[:, 2] + output[:, 3], dy[:, 2] + output[:, 4],
                output[:, 5], output[:, 6],
            ), dim=1)
            density = 0.5 * ((strains @ self.thickness_stiffness) * strains).sum(1, keepdim=True)
            w0 = output[:, 2:3]
            if return_density:
                return density, w0
            area = self.Lx * self.Ly
            if is_gauss:
                return ((density - self.q0 * w0) * self.gauss_wts[:, None]).sum() * (area / 4)
            return (density - self.q0 * w0).mean() * area

    def scan_adaptive_regions(self):
        scan_x = torch.linspace(-0.99, 0.99, 80, device=DEVICE)
        scan_y = torch.linspace(-0.99, 0.99, 80, device=DEVICE)
        gx, gy = torch.meshgrid(scan_x, scan_y, indexing='ij')
        points = torch.stack((gx.flatten(), gy.flatten()), dim=1)
        density, _ = self.compute_energy(points, return_density=True)
        count = min(self.adaptive_centers_count, density.numel())
        indices = torch.topk(density.squeeze(), count).indices
        self.adaptive_points = points[indices].detach()

    def generate_samples(self, count):
        adaptive_count = (
            int(count * self.adaptive_fraction) if self.adaptive_points is not None else 0
        )
        sobol_points = self.sobol.draw(
            count - adaptive_count, dtype=self.props.dtype
        ).to(DEVICE) * 2 - 1
        if adaptive_count == 0:
            return sobol_points
        indices = torch.randint(
            0, len(self.adaptive_points), (adaptive_count,), device=self.adaptive_points.device
        )
        noise = torch.randn(adaptive_count, 2, device=DEVICE) * self.adaptive_noise
        adaptive = (self.adaptive_points[indices] + noise).clamp(-0.99, 0.99)
        return torch.cat((sobol_points, adaptive), dim=0)


def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def read_references():
    grid_x, grid_y = np.meshgrid(
        np.linspace(-LX / 2, LX / 2, 201),
        np.linspace(-LY / 2, LY / 2, 101),
        indexing='ij',
    )
    references = {}
    hashes = {}
    for key, filename in zip(('u', 'v', 'w'), ('5X.xlsx', '5Y.xlsx', '5Z.xlsx')):
        path = ROOT / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing FE reference: {path}')
        print(f'读取有限元参考解: {filename}', flush=True)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        values = pd.read_excel(path).to_numpy(dtype=float)
        source_points = values[:, :2] / 1000 - np.array([LX / 2, LY / 2])
        interpolated = griddata(source_points, values[:, -1], (grid_x, grid_y), method='cubic')
        missing = ~np.isfinite(interpolated)
        if missing.any():
            interpolated[missing] = griddata(
                source_points,
                values[:, -1],
                (grid_x[missing], grid_y[missing]),
                method='nearest',
            )
        references[key] = interpolated
    return references, hashes


def predict(model):
    xi = torch.linspace(-1, 1, 201, device=DEVICE)
    eta = torch.linspace(-1, 1, 101, device=DEVICE)
    gx, gy = torch.meshgrid(xi, eta, indexing='ij')
    points = torch.stack((gx.flatten(), gy.flatten()), dim=1)
    with torch.no_grad():
        output = model.net(points) * model.scale
        phi_x, phi_y = model.get_phi(model.z_layers[-1:])
        fields = (
            output[:, 0] + TOTAL_THICKNESS / 2 * output[:, 3] + phi_x * output[:, 5],
            output[:, 1] + TOTAL_THICKNESS / 2 * output[:, 4] + phi_y * output[:, 6],
            output[:, 2],
        )
    return {
        key: field.reshape(201, 101).cpu().numpy() * 1e5
        for key, field in zip(('u', 'v', 'w'), fields)
    }


def calculate_metrics(fields, references):
    metrics = {}
    grid_x, grid_y = np.meshgrid(
        np.linspace(-LX / 2, LX / 2, 201),
        np.linspace(-LY / 2, LY / 2, 101),
        indexing='ij',
    )
    for key in ('u', 'v', 'w'):
        difference = fields[key] - references[key]
        absolute = np.abs(difference)
        peak = np.unravel_index(np.argmax(absolute), difference.shape)
        max_error = float(absolute[peak])
        max_reference = float(np.max(np.abs(references[key])))
        metrics[key] = {
            'max_abs_error_10um': max_error,
            'max_abs_fe_10um': max_reference,
            'relative_max_percent': 100 * max_error / max(max_reference, 1e-15),
            'relative_l2_percent': 100 * float(
                np.linalg.norm(difference) / max(np.linalg.norm(references[key]), 1e-15)
            ),
            'peak_x_m': float(grid_x[peak]),
            'peak_y_m': float(grid_y[peak]),
        }
    return metrics


def diagnostics(model):
    model.zero_grad(set_to_none=True)
    energy = model.compute_energy(model.gauss_pts, is_gauss=True)
    normalized = energy / model.energy_normalizer
    normalized.backward()
    gradient = torch.sqrt(sum(
        parameter.grad.detach().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    )).item()
    model.zero_grad(set_to_none=True)
    return {
        'normalized_energy': float(normalized.detach()),
        'parameter_gradient_norm': gradient,
    }


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train(args, references):
    model = RZTPINN(args).to(DEVICE)
    model.prepare_stiffness()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.adam_epochs), eta_min=args.adam_min_lr
    )
    scan_count = 0
    log_interval = max(1, min(500, args.adam_epochs // 10))
    synchronize()
    adam_started = time.perf_counter()
    progress = trange(args.adam_epochs, desc='Adam Training')
    for epoch in progress:
        if (epoch >= args.adaptive_start
                and (epoch - args.adaptive_start) % args.adaptive_interval == 0):
            model.scan_adaptive_regions()
            scan_count += 1
        points = model.generate_samples(args.samples)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_energy(points) / model.energy_normalizer
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        optimizer.step()
        scheduler.step()
        if epoch % log_interval == 0 or epoch + 1 == args.adam_epochs:
            record = {
                'epoch': epoch + 1,
                'normalized_energy': float(loss.detach()),
                'learning_rate': scheduler.get_last_lr()[0],
                'gradient_norm_before_clipping': gradient_norm,
                'adaptive_active': model.adaptive_points is not None,
                'hotspot_weight': args.adaptive_fraction if model.adaptive_points is not None else 0.0,
            }
            progress.set_postfix({
                'Energy': f'{record["normalized_energy"]:.4f}',
                'LR': f'{record["learning_rate"]:.2e}',
                'Adaptive': f'{record["hotspot_weight"]:.2f}',
            })
    synchronize()
    adam_seconds = time.perf_counter() - adam_started
    adam_fields = predict(model)
    adam_metrics = calculate_metrics(adam_fields, references)
    adam_diagnostics = diagnostics(model)
    best_score = max(item['relative_max_percent'] for item in adam_metrics.values())
    best_state = snapshot(model)
    best_iterations = args.adam_epochs
    best_lbfgs_iterations = 0
    best_metrics = adam_metrics
    best_diagnostics = adam_diagnostics
    print(f'Adam训练完成: {args.adam_epochs}轮, {adam_seconds:.2f}s')

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=args.block,
        history_size=args.history,
        tolerance_grad=1e-11,
        tolerance_change=1e-13,
        line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_energy(model.gauss_pts, is_gauss=True) / model.energy_normalizer
        loss.backward()
        return loss

    print('\nL-BFGS Refinement...')
    stale_blocks = 0
    lbfgs_started = time.perf_counter()
    completed = 0
    for start in range(0, args.iters, args.block):
        block_iterations = min(args.block, args.iters - start)
        optimizer.param_groups[0]['max_iter'] = block_iterations
        optimizer.param_groups[0]['max_eval'] = block_iterations * 2
        optimizer.step(closure)
        completed = start + block_iterations
        fields = predict(model)
        metrics = calculate_metrics(fields, references)
        diagnostic = diagnostics(model)
        score = max(item['relative_max_percent'] for item in metrics.values())
        record = {
            'iterations': args.adam_epochs + completed,
            'lbfgs_iterations': completed,
            'metrics': metrics,
            'diagnostics': diagnostic,
            'elapsed_seconds': time.perf_counter() - lbfgs_started,
        }
        current_state = snapshot(model)
        if score < best_score:
            best_score = score
            best_state = current_state
            best_iterations = args.adam_epochs + completed
            best_lbfgs_iterations = completed
            best_metrics = metrics
            best_diagnostics = diagnostic
            stale_blocks = 0
        else:
            stale_blocks += 1
        should_stop = bool(args.patience and stale_blocks >= args.patience)
        if completed % 300 == 0 or completed == args.iters or should_stop:
            errors = ', '.join(
                f'{key}={metrics[key]["relative_max_percent"]:.4f}%'
                for key in ('u', 'v', 'w')
            )
            print(
                f'L-BFGS {completed}/{args.iters}: '
                f'Energy={diagnostic["normalized_energy"]:.8e}, '
                f'{errors}, Time={record["elapsed_seconds"]:.1f}s',
                flush=True,
            )
        if should_stop:
            print(
                f'Early stop: no improvement for {stale_blocks} evaluation blocks; '
                'retaining the best model.',
                flush=True,
            )
            break
    synchronize()
    lbfgs_seconds = time.perf_counter() - lbfgs_started
    model.load_state_dict(best_state)
    model.prepare_stiffness()
    return model, {
        'adam_seconds': adam_seconds,
        'lbfgs_seconds': lbfgs_seconds,
        'actual_lbfgs_iterations': completed,
        'selected_iterations': best_iterations,
        'selected_lbfgs_iterations': best_lbfgs_iterations,
        'selected_metrics': best_metrics,
        'selected_diagnostics': best_diagnostics,
        'adaptive_scans': scan_count,
    }


def plot_postprocess(model, fields):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    model.eval()
    nx, ny = fields['w'].shape
    xi = torch.linspace(-1, 1, nx, device=DEVICE)
    eta = torch.linspace(-1, 1, ny, device=DEVICE)
    gx, gy = torch.meshgrid(xi, eta, indexing='ij')
    points = torch.stack((gx.flatten(), gy.flatten()), dim=1)
    with torch.no_grad():
        output = model.net(points) * model.scale
        center = model.net(torch.zeros((1, 2), device=DEVICE)) * model.scale
    x = gx.cpu().numpy() * LX / 2
    y = gy.cpu().numpy() * LY / 2
    unit = r'$10\,\mu\mathrm{m}$'
    panels = [
        (fields['u'], r'$u_{top}$', unit),
        (fields['v'], r'$v_{top}$', unit),
        (fields['w'], r'$w_0$', unit),
    ]
    for index, title in zip(
            (3, 4, 5, 6),
            (r'$\theta_x$', r'$\theta_y$', r'$\psi_x$', r'$\psi_y$')):
        panels.append((output[:, index].reshape(nx, ny).cpu().numpy(), title, 'rad'))
    figure, axes = plt.subplots(2, 4, figsize=(22, 10))
    figure.canvas.manager.set_window_title('RZT fields and centerline')
    for axis, (values, title, panel_unit) in zip(axes.flat, panels):
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-15:
            low, high = low - 1e-15, high + 1e-15
        contour = axis.contourf(x, y, values, levels=np.linspace(low, high, 50), cmap='jet')
        colorbar = figure.colorbar(
            contour,
            ax=axis,
            ticks=np.linspace(low, high, 7),
            format=FormatStrFormatter('%.3e'),
        )
        colorbar.set_label(panel_unit)
        axis.set_title(f'{title} ({panel_unit})')
        axis.set_aspect('equal')
    centerline = fields['w'][:, ny // 2]
    axes[1, 3].plot(x[:, ny // 2], centerline, 'b-')
    axes[1, 3].set_title(f'Centerline w (max |w|={np.abs(centerline).max():.3f} {unit})')
    axes[1, 3].grid(True)
    figure.tight_layout()

    layers = model.z_layers.detach().cpu().numpy()
    z_values = np.unique(np.concatenate([
        np.linspace(lower, upper, 51)
        for lower, upper in zip(layers[:-1], layers[1:])
    ]))
    z = torch.tensor(z_values, device=DEVICE)
    with torch.no_grad():
        phi_x, phi_y = model.get_phi(z)
        profiles = (
            (center[0, 0] + z * center[0, 3] + phi_x * center[0, 5]) * 1e5,
            (center[0, 1] + z * center[0, 4] + phi_y * center[0, 6]) * 1e5,
            torch.ones_like(z) * center[0, 2] * 1e5,
        )
    thickness_figure, thickness_axes = plt.subplots(1, 3, figsize=(15, 5))
    thickness_figure.canvas.manager.set_window_title('RZT thickness profiles at plate center')
    for axis, values, key, color in zip(
            thickness_axes, profiles, ('u', 'v', 'w'), ('b', 'r', 'g')):
        axis.plot(values.cpu().numpy(), z_values * 1000, color=color)
        axis.set_title(f'{key}(z) at center ({unit})')
        axis.grid(True)
    thickness_figure.tight_layout()
    return [figure, thickness_figure]


def plot_comparison(fields, references):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    grid_x, grid_y = np.meshgrid(
        np.linspace(-LX / 2, LX / 2, 201),
        np.linspace(-LY / 2, LY / 2, 101),
        indexing='ij',
    )
    figure, axes = plt.subplots(
        3, 3, figsize=(18, 10.8), gridspec_kw={'wspace': 0.25, 'hspace': 0.35}
    )
    for row, (key, label) in enumerate(zip(
            ('u', 'v', 'w'), (r'$u_{top}$', r'$v_{top}$', r'$w_0$'))):
        values_by_column = (
            ('PINN', fields[key]),
            ('FE', references[key]),
            ('Difference', fields[key] - references[key]),
        )
        for column, (kind, values) in enumerate(values_by_column):
            low, high = float(values.min()), float(values.max())
            if high - low < 1e-12:
                low, high = low - 1e-12, high + 1e-12
            axis = axes[row, column]
            contour = axis.contourf(
                grid_x,
                grid_y,
                values,
                levels=np.linspace(low, high, 100),
                cmap='coolwarm' if column == 2 else 'jet',
            )
            color_axis = make_axes_locatable(axis).append_axes('right', size='3%', pad=0.15)
            colorbar = figure.colorbar(
                contour,
                cax=color_axis,
                ticks=np.linspace(low, high, 5),
                format=FormatStrFormatter('%.3f'),
            )
            colorbar.ax.tick_params(labelsize=17)
            colorbar.set_label(r'$10\,\mu\mathrm{m}$', fontsize=17)
            axis.set_title(f'{kind}: {label} ' + r'($10\,\mu\mathrm{m}$)', fontsize=17)
            axis.set_aspect('equal')
            axis.set_xticks([])
            axis.set_yticks([])
    return figure


def parse_solver_args(argv=None):
    parser = argparse.ArgumentParser(description='Random-initialization 5 MPa honeycomb RZT-PINN solver.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for random network initialization and sampling.',
    )
    parser.add_argument('--fourier-dim', type=int, default=128)
    parser.add_argument('--fourier-scale', type=float, default=0.2)
    parser.add_argument('--in-plane-power', type=float, default=1.0)
    parser.add_argument('--pinn-w-power', type=float, default=2.0)
    parser.add_argument('--adam-epochs', type=int, default=8000)
    parser.add_argument('--adam-lr', type=float, default=2e-4)
    parser.add_argument('--adam-min-lr', type=float, default=1e-6)
    parser.add_argument(
        '--samples', type=int, default=4000,
        help='Total collocation points generated per Adam epoch.',
    )
    parser.add_argument(
        '--adaptive-fraction', type=float, default=0.40,
        help='Energy-adaptive share after warm-up; the remaining share uses Sobol points.',
    )
    parser.add_argument('--adaptive-centers', type=int, default=1200)
    parser.add_argument('--adaptive-noise', type=float, default=0.03)
    parser.add_argument(
        '--adaptive-start', type=int, default=4000,
        help='Sobol-only warm-up epochs before hybrid sampling continues to the end of Adam.',
    )
    parser.add_argument('--adaptive-interval', type=int, default=50)
    parser.add_argument('--iters', type=int, default=1500)
    parser.add_argument('--block', type=int, default=50)
    parser.add_argument('--gauss-order', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--history', type=int, default=300)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--label', default='refined')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args(argv)
    if (args.seed is not None and args.seed < 0
            or args.fourier_dim < 2 or args.fourier_dim % 2
            or args.fourier_scale <= 0
            or args.in_plane_power < 1 or args.pinn_w_power < 1
            or args.adam_epochs < 2
            or args.adam_lr <= 0 or args.adam_min_lr <= 0
            or args.adam_min_lr > args.adam_lr
            or args.samples < 1
            or not 0 < args.adaptive_fraction < 1
            or args.adaptive_centers < 1 or args.adaptive_noise < 0
            or not 1 <= args.adaptive_start < args.adam_epochs
            or args.adaptive_interval < 1
            or args.iters < 1 or args.block < 1
            or args.gauss_order < 2 or args.lr <= 0
            or args.history < 1 or args.patience < 0
            or not args.label or args.label in ('.', '..')
            or Path(args.label).name != args.label):
        parser.error('Invalid network, sampling, quadrature, or optimizer settings.')
    return args


def main():
    args = parse_solver_args()
    validate_stiffness()
    references, _ = read_references()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, _ = train(args, references)
    fields = predict(model)
    metrics = calculate_metrics(fields, references)
    figures = []
    if not args.no_plot:
        figures = plot_postprocess(model, fields)
        figures.append(plot_comparison(fields, references))
    print('\n三分量相对最大误差: ' + ', '.join(
        f'{key}={metrics[key]["relative_max_percent"]:.6f}%'
        for key in ('u', 'v', 'w')
    ), flush=True)
    if args.no_show:
        for figure in figures:
            plt.close(figure)
    elif figures:
        print('关闭图窗后程序结束。', flush=True)
        plt.show()


if __name__ == '__main__':
    main()
