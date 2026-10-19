import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from scipy.interpolate import griddata
from tqdm import trange
from torch.quasirandom import SobolEngine

# ==========================================
# 1. 基础配置与物理参数
# ==========================================
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# 几何尺寸与载荷条件
Lx, Ly, h_total, q0 = 0.1, 0.04, 0.002, 1.0e6
REFERENCE_LOAD = 100.0e6
DISPLACEMENT_COMPARISON_UNIT = '0.01 mm'
DISPLACEMENT_COMPARISON_SCALE = 1.0 / (0.01 * 1.0e-3)
ADAPTIVE_START = 3000
ADAPTIVE_INTERVAL = 50
ADAPTIVE_FRACTION = 0.40
ADAPTIVE_CENTERS = 1200
ADAPTIVE_NOISE = 0.03

# 分层参数与材料属性
h_k_np = np.array([0.0005, 0.001, 0.0005])
z_layers_np = np.array([-0.001, -0.0005, 0.0005, 0.001])

# 材料属性矩阵 [E, nu, G]
props_np = np.array([
    [1.93e11, 0.31, 7.37e10],
    [2.00e11, 0.30, 7.69e10],
    [1.93e11, 0.31, 7.37e10]
])

# 计算 RZT 剪切修正系数
G_k_np = props_np[:, 2]
Gavg_np = h_total / np.sum(h_k_np / G_k_np)
beta_x_np = Gavg_np / G_k_np - 1.0
beta_y_np = beta_x_np.copy()


# ==========================================
# 2. 神经网络架构
# ==========================================
class FourierEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, scale=0.2):
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, out_dim // 2, dtype=torch.float64) * scale)

    def forward(self, x):
        x_proj = (x @ self.B) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class RZT_ResNet(nn.Module):
    def __init__(self, n_in=2, n_out=7):
        super().__init__()
        self.encoder = FourierEncoding(n_in, 128, scale=0.2)
        self.fc1 = nn.Linear(128, 128)
        self.res = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.fc_out = nn.Linear(128, n_out)
        self.register_buffer('weights', torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]))

    def forward(self, x):
        xi, eta = x[:, 0:1], x[:, 1:2]
        symmetric_coordinates = torch.cat([xi ** 2, eta ** 2], dim=-1)
        h = self.encoder(symmetric_coordinates)
        h = torch.tanh(self.fc1(h))
        h = h + self.res(h)
        out = self.fc_out(h) * self.weights

        dist = (1.0 - xi ** 2) * (1.0 - eta ** 2)
        symmetry = torch.cat([xi, eta, torch.ones_like(xi), xi, eta, xi, eta], dim=-1)
        boundary = torch.cat([
            dist, dist, dist ** 2, dist, dist, dist, dist
        ], dim=-1)
        return out * symmetry * boundary


# ==========================================
# 3. RZT 能量法模型
# ==========================================
class RZT_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lx, self.Ly, self.q0 = Lx, Ly, q0
        self.displacement_scale = 0.01 * (self.q0 / REFERENCE_LOAD)
        self.rotation_scale = self.displacement_scale / min(self.Lx, self.Ly)

        self.register_buffer('props', torch.tensor(props_np, dtype=torch.float64))
        self.register_buffer('z_layers', torch.tensor(z_layers_np, dtype=torch.float64))
        self.register_buffer('h_k', torch.tensor(h_k_np, dtype=torch.float64))
        self.register_buffer('beta_x', torch.tensor(beta_x_np, dtype=torch.float64))
        self.register_buffer('beta_y', torch.tensor(beta_y_np, dtype=torch.float64))
        self.register_buffer('output_scales', torch.tensor([
            self.displacement_scale,
            self.displacement_scale,
            self.displacement_scale,
            self.rotation_scale,
            self.rotation_scale,
            self.rotation_scale,
            self.rotation_scale
        ], dtype=torch.float64))

        self.net = RZT_ResNet().to(device)
        self.sobol = SobolEngine(dimension=2, scramble=True)
        self.adaptive_points = None
        self.energy_scale = self.q0 * self.displacement_scale * self.Lx * self.Ly
        self._setup_integrators()

    def predict_fields(self, xi_eta):
        return self.net(xi_eta) * self.output_scales

    def _setup_integrators(self):
        gp = torch.tensor([-0.77459667, 0.0, 0.77459667], dtype=torch.float64)
        gw = torch.tensor([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=torch.float64)
        all_z, all_w, all_k = [], [], []

        for k in range(len(h_k_np)):
            zc = (self.z_layers[k] + self.z_layers[k + 1]) / 2.0
            hh = self.h_k[k] / 2.0
            all_z.append(zc + gp * hh)
            all_w.append(gw * hh)
            all_k.append(torch.full_like(gp, k, dtype=torch.long))

        self.register_buffer('int_z', torch.cat(all_z))
        self.register_buffer('int_w', torch.cat(all_w))
        self.register_buffer('int_k', torch.cat(all_k))

        ng = 50
        pts_leg, wts_leg = np.polynomial.legendre.leggauss(ng)
        gx, gy = torch.meshgrid(torch.tensor(pts_leg, dtype=torch.float64),
                                torch.tensor(pts_leg, dtype=torch.float64), indexing='ij')
        self.register_buffer('gauss_pts', torch.stack([gx.flatten(), gy.flatten()], dim=1))
        wt_t = torch.tensor(wts_leg, dtype=torch.float64)
        self.register_buffer('gauss_wts', (wt_t.unsqueeze(0) * wt_t.unsqueeze(1)).flatten())

    def get_phi(self, z_vals):
        phi_x, phi_y = torch.zeros_like(z_vals), torch.zeros_like(z_vals)
        for k in range(len(self.h_k)):
            mask = (z_vals >= self.z_layers[k] - 1e-9) & (z_vals <= self.z_layers[k + 1] + 1e-9)
            if mask.any():
                pk_x = torch.sum(self.beta_x[:k] * self.h_k[:k]) if k > 0 else 0.0
                pk_y = torch.sum(self.beta_y[:k] * self.h_k[:k]) if k > 0 else 0.0
                phi_x[mask] = pk_x + self.beta_x[k] * (z_vals[mask] - self.z_layers[k])
                phi_y[mask] = pk_y + self.beta_y[k] * (z_vals[mask] - self.z_layers[k])
        return phi_x, phi_y

    def compute_energy(self, xi_eta, is_lbfgs=False, return_density=False):
        needs_graph = not return_density
        xi_eta = xi_eta.detach().requires_grad_(True)
        out = self.predict_fields(xi_eta)
        u0, v0, w0, tx, ty, px, py = out.split(1, dim=-1)

        def grad(f):
            return torch.autograd.grad(f, xi_eta, torch.ones_like(f), create_graph=needs_graph, retain_graph=True)[0]

        gu, gv, gw_g, gtx, gty, gpx, gpy = map(grad, [u0, v0, w0, tx, ty, px, py])
        jx, jy = 2.0 / self.Lx, 2.0 / self.Ly
        ux, uy, vx, vy, wx, wy = gu[:, 0:1] * jx, gu[:, 1:2] * jy, gv[:, 0:1] * jx, gv[:, 1:2] * jy, gw_g[:,
                                                                                                     0:1] * jx, gw_g[:,
                                                                                                                1:2] * jy
        tx_x, tx_y, ty_x, ty_y = gtx[:, 0:1] * jx, gtx[:, 1:2] * jy, gty[:, 0:1] * jx, gty[:, 1:2] * jy
        px_x, px_y, py_x, py_y = gpx[:, 0:1] * jx, gpx[:, 1:2] * jy, gpy[:, 0:1] * jx, gpy[:, 1:2] * jy

        phi_x, phi_y = self.get_phi(self.int_z)
        U_density = torch.zeros(xi_eta.size(0), 1, device=device)

        for i in range(len(self.int_z)):
            z, w_z, k = self.int_z[i], self.int_w[i], self.int_k[i]
            E, nu, G = self.props[k]
            Q11, Q12, Q66 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), G
            Q44, Q55 = G, G

            ex, ey = ux + z * tx_x + phi_x[i] * px_x, vy + z * ty_y + phi_y[i] * py_y
            gxy = (uy + vx) + z * (tx_y + ty_x) + phi_x[i] * px_y + phi_y[i] * py_x
            gxz, gyz = (wx + tx) + self.beta_x[k] * px, (wy + ty) + self.beta_y[k] * py
            U_density += 0.5 * (Q11 * ex ** 2 + 2 * Q12 * ex * ey + Q11 * ey ** 2 +
                                Q66 * gxy ** 2 + Q55 * gxz ** 2 + Q44 * gyz ** 2) * w_z

        if return_density: return U_density, w0

        Area = self.Lx * self.Ly
        if is_lbfgs:
            wts = self.gauss_wts.unsqueeze(1)
            return torch.sum(U_density * wts) * (Area / 4.0) - torch.sum(w0 * wts) * self.q0 * (Area / 4.0)
        return torch.mean(U_density) * Area - torch.mean(w0 * self.q0) * Area

    def generate_samples(self, n=4000):
        adaptive_count = (
            int(n * ADAPTIVE_FRACTION) if self.adaptive_points is not None else 0
        )
        sobol_points = self.sobol.draw(n - adaptive_count).to(device) * 2.0 - 1.0
        if adaptive_count == 0:
            return sobol_points

        indices = torch.randint(
            0, len(self.adaptive_points), (adaptive_count,), device=device
        )
        noise = torch.randn(adaptive_count, 2, device=device) * ADAPTIVE_NOISE
        adaptive = (self.adaptive_points[indices] + noise).clamp(-0.99, 0.99)
        return torch.cat((sobol_points, adaptive), dim=0)

    def scan_adaptive_regions(self):
        scan_x = torch.linspace(-0.99, 0.99, 80, device=device)
        scan_y = torch.linspace(-0.99, 0.99, 80, device=device)
        gx, gy = torch.meshgrid(scan_x, scan_y, indexing='ij')
        points = torch.stack((gx.flatten(), gy.flatten()), dim=1)
        density, _ = self.compute_energy(points, return_density=True)
        count = min(ADAPTIVE_CENTERS, density.numel())
        indices = torch.topk(density.squeeze(), count).indices
        self.adaptive_points = points[indices].detach()


# ==========================================
# 4. 训练与精调
# ==========================================
def train():
    model = RZT_PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=7000, eta_min=2e-5
    )
    pbar = trange(7000, desc="Adam Stage")

    for epoch in pbar:
        if (epoch >= ADAPTIVE_START
                and (epoch - ADAPTIVE_START) % ADAPTIVE_INTERVAL == 0):
            model.scan_adaptive_regions()

        pts = model.generate_samples(n=6000)
        optimizer.zero_grad()
        loss = model.compute_energy(pts)
        scaled_loss = loss / model.energy_scale
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            pbar.set_postfix({
                'Energy': f"{scaled_loss.item():.6f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}",
                'Adaptive': f"{ADAPTIVE_FRACTION if model.adaptive_points is not None else 0.0:.2f}"
            })

    print("\nL-BFGS Refinement...")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=1000,
        history_size=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        l = model.compute_energy(model.gauss_pts, is_lbfgs=True)
        scaled_l = l / model.energy_scale
        scaled_l.backward()
        return scaled_l

    lbfgs.step(closure)
    return model


# ==========================================
# 5. 后处理与对比模块
# ==========================================
def postprocess(model):
    print("\n--- 可视化场变量 ---")
    model.eval()
    cpt = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=device)
    with torch.no_grad():
        out_c = model.predict_fields(cpt)
    w_center = out_c[0, 2].item() * DISPLACEMENT_COMPARISON_SCALE

    Nx, Ny = 101, 51
    xi = torch.linspace(-1, 1, Nx, device=device)
    eta = torch.linspace(-1, 1, Ny, device=device)
    XI, ETA = torch.meshgrid(xi, eta, indexing='ij')
    pts = torch.stack([XI.flatten(), ETA.flatten()], dim=1)

    with torch.no_grad():
        out = model.predict_fields(pts)

    u0 = out[:, 0].reshape(Nx, Ny).cpu().numpy()
    v0 = out[:, 1].reshape(Nx, Ny).cpu().numpy()
    w0 = out[:, 2].reshape(Nx, Ny).cpu().numpy() * DISPLACEMENT_COMPARISON_SCALE
    tx = out[:, 3].reshape(Nx, Ny).cpu().numpy()
    ty = out[:, 4].reshape(Nx, Ny).cpu().numpy()
    px = out[:, 5].reshape(Nx, Ny).cpu().numpy()
    py = out[:, 6].reshape(Nx, Ny).cpu().numpy()

    u_top = (u0 + (h_total / 2.0) * tx) * DISPLACEMENT_COMPARISON_SCALE
    v_top = (v0 + (h_total / 2.0) * ty) * DISPLACEMENT_COMPARISON_SCALE
    X, Y = XI.cpu().numpy() * Lx / 2, ETA.cpu().numpy() * Ly / 2

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fields = [(u_top, rf'$u_{{top}}$ ({DISPLACEMENT_COMPARISON_UNIT})'),
              (v_top, rf'$v_{{top}}$ ({DISPLACEMENT_COMPARISON_UNIT})'),
              (w0, rf'$w_0$ ({DISPLACEMENT_COMPARISON_UNIT})'),
              (tx, r'$\theta_x$ (rad)'), (ty, r'$\theta_y$ (rad)'),
              (px, r'$\psi_x$ (rad)'), (py, r'$\psi_y$ (rad)')]

    for idx, (field, title) in enumerate(fields):
        ax = axes[idx // 4, idx % 4]
        cf = ax.contourf(X, Y, field, 50, cmap='jet')
        cbar = plt.colorbar(cf, ax=ax, format=FormatStrFormatter('%.3e'))
        cbar.locator = LinearLocator(numticks=5)  # 精确 5 个分段刻度
        cbar.update_ticks()
        ax.set_title(title)
        ax.set_aspect('equal')

    ax_last = axes[1, 3]
    ax_last.plot(X[:, Ny // 2], w0[:, Ny // 2], 'b-')
    ax_last.set_title(f'Centerline w (max={np.max(w0):.4f} {DISPLACEMENT_COMPARISON_UNIT})')
    ax_last.grid(True)
    plt.tight_layout()
    plt.show()


def compare_with_simulation(model):
    print("\n" + "=" * 50)
    print("执行 PINN 与 FEM 仿真对比")
    print("=" * 50)

    model.eval()
    Nx, Ny = 201, 101
    xi = torch.linspace(-1, 1, Nx, device=device)
    eta = torch.linspace(-1, 1, Ny, device=device)
    XI, ETA = torch.meshgrid(xi, eta, indexing='ij')
    pts = torch.stack([XI.flatten(), ETA.flatten()], dim=1)

    with torch.no_grad():
        out = model.predict_fields(pts)

    u0 = out[:, 0].reshape(Nx, Ny).cpu().numpy()
    v0 = out[:, 1].reshape(Nx, Ny).cpu().numpy()
    w0 = out[:, 2].reshape(Nx, Ny).cpu().numpy()
    tx = out[:, 3].reshape(Nx, Ny).cpu().numpy()
    ty = out[:, 4].reshape(Nx, Ny).cpu().numpy()

    u_pinn = (u0 + (h_total / 2.0) * tx) * DISPLACEMENT_COMPARISON_SCALE
    v_pinn = (v0 + (h_total / 2.0) * ty) * DISPLACEMENT_COMPARISON_SCALE
    w_pinn = w0 * DISPLACEMENT_COMPARISON_SCALE
    X_grid, Y_grid = XI.cpu().numpy() * Lx / 2, ETA.cpu().numpy() * Ly / 2

    def load_and_interpolate(filename, grid_x, grid_y):
        try:
            df = pd.read_excel(filename)
            data = df.values
            pts_sim = np.column_stack(((data[:, 0] / 1000.0) - (Lx / 2.0), (data[:, 1] / 1000.0)))
            vals_sim = data[:, -1]
            return griddata(pts_sim, vals_sim, (grid_x, grid_y), method='cubic')
        except Exception:
            return np.zeros_like(grid_x)

    u_sim = load_and_interpolate('CX1up.xlsx', X_grid, Y_grid)
    v_sim = load_and_interpolate('CY1up.xlsx', X_grid, Y_grid)
    w_sim = load_and_interpolate('CZ1up.xlsx', X_grid, Y_grid)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(3, 3, figsize=(18, 8), gridspec_kw={'wspace': 0.25, 'hspace': 0.15})
    plot_data = [(u_pinn, u_sim, 'u_top'), (v_pinn, v_sim, 'v_top'), (w_pinn, w_sim, 'w_0')]

    for i, (pinn, sim, title) in enumerate(plot_data):
        error = pinn - sim
        valid = np.isfinite(pinn) & np.isfinite(sim)
        relative_l2 = np.linalg.norm(error[valid]) / np.linalg.norm(sim[valid])
        rmse = np.sqrt(np.mean(error[valid] ** 2))
        max_abs_error = np.max(np.abs(error[valid]))
        print(
            f'{title}: relative L2={relative_l2:.3%}, '
            f'RMSE={rmse:.6f} {DISPLACEMENT_COMPARISON_UNIT}, '
            f'max error={max_abs_error:.6f} {DISPLACEMENT_COMPARISON_UNIT}'
        )

        row = [pinn, sim, error]
        for j, data in enumerate(row):
            cmap = 'jet' if j < 2 else 'coolwarm'
            cf = axes[i, j].contourf(X_grid, Y_grid, data, 100, cmap=cmap)
            divider = make_axes_locatable(axes[i, j])
            cax = divider.append_axes("right", size="3%", pad=0.15)
            cbar = plt.colorbar(cf, cax=cax, format=FormatStrFormatter('%.3f'))
            cbar.set_label(DISPLACEMENT_COMPARISON_UNIT)
            cbar.set_ticks(np.linspace(np.nanmin(data), np.nanmax(data), 5))
            cbar.ax.tick_params(labelsize=17)
            cbar.update_ticks()
            axes[i, j].set_aspect('equal')

            axes[i, j].set_xticks([])  # 隐藏x轴刻度
            axes[i, j].set_yticks([])  # 隐藏y轴刻度
            axes[i, j].set_title(f"{title} - {'PINN' if j == 0 else 'FEM' if j == 1 else 'Error'}")

    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    trained_model = train()
    postprocess(trained_model)
    compare_with_simulation(trained_model)
