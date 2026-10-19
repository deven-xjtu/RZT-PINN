import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata
from tqdm import trange
from torch.quasirandom import SobolEngine
import random

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RANDOM_SEED = 42

# ==========================================
# 1. 基础配置与材料参数
# ==========================================
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

# 几何与载荷（q0 使用 Pa；120 kPa = 120e3 Pa）
Lx, Ly, q0 = 0.05, 0.03, 120e3
num_layers = 5
h_layer = 0.0005
h_total = h_layer * num_layers
z_layers_np = np.linspace(-h_total / 2, h_total / 2, num_layers + 1)
h_k_np = np.diff(z_layers_np)
angles = [0, 45, 90, 45, 0]

# 材料属性
E1, E2 = 1.21e11, 8.6e9
nu12, G12, G23, G13 = 0.27, 4.7e9, 3.1e9, 4.7e9


def get_Q_bar(E1, E2, nu12, G12, G13, G23, theta_deg):
    nu21 = nu12 * E2 / E1
    f = 1.0 - nu12 * nu21
    Q11, Q22, Q12 = E1 / f, E2 / f, nu12 * E2 / f
    Q66, Q55, Q44 = G12, G13, G23
    theta = np.radians(theta_deg)
    m, n = np.cos(theta), np.sin(theta)
    return np.array([
        Q11 * m ** 4 + 2 * (Q12 + 2 * Q66) * m ** 2 * n ** 2 + Q22 * n ** 4,
        Q11 * n ** 4 + 2 * (Q12 + 2 * Q66) * m ** 2 * n ** 2 + Q22 * m ** 4,
        (Q11 + Q22 - 4 * Q66) * m ** 2 * n ** 2 + Q12 * (m ** 4 + n ** 4),
        (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m ** 2 * n ** 2 + Q66 * (m ** 4 + n ** 4),
        (Q11 - Q12 - 2 * Q66) * m ** 3 * n + (Q12 - Q22 + 2 * Q66) * m * n ** 3,
        (Q11 - Q12 - 2 * Q66) * m * n ** 3 + (Q12 - Q22 + 2 * Q66) * m ** 3 * n,
        Q55 * m ** 2 + Q44 * n ** 2,
        Q44 * m ** 2 + Q55 * n ** 2,
        (Q55 - Q44) * m * n
    ])


props_np = np.vstack([get_Q_bar(E1, E2, nu12, G12, G13, G23, a) for a in angles])
Gxz_k, Gyz_k = props_np[:, 6], props_np[:, 7]
Gavg_x, Gavg_y = h_total / np.sum(h_k_np / Gxz_k), h_total / np.sum(h_k_np / Gyz_k)
beta_x_np, beta_y_np = Gavg_x / Gxz_k - 1.0, Gavg_y / Gyz_k - 1.0

DISPLACEMENT_SCALE = 0.012e-3
ENERGY_NORMALIZATION = 1.0
LOAD_RAMP_EPOCHS = 2000
LBFGS_MAX_ITER = 300


# ==========================================
# 2. 网络架构
# ==========================================
class FourierEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, scale=0.2):
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, out_dim // 2, dtype=torch.float64) * scale)

    def forward(self, x):
        x_proj = (x @ self.B) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh())

    def forward(self, x): return x + self.net(x)


class RZT_ResNet(nn.Module):
    def __init__(self, n_in=2, n_out=7):
        super().__init__()
        hidden = 128
        self.encoder = FourierEncoding(n_in, hidden, scale=0.2)
        self.fc_in = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(2)])
        self.fc_out = nn.Linear(hidden, n_out)
        self.register_buffer("weights", torch.tensor([1.0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0]))

    def forward(self, x):
        h = self.encoder(x)
        h = torch.tanh(self.fc_in(h))
        for b in self.blocks: h = b(h)
        out = self.fc_out(h) * self.weights
        xi, eta = x[:, 0:1], x[:, 1:2]
        dist = (1.0 - xi ** 2) * (1.0 - eta ** 2)
        return torch.cat([out[:, 0:2] * dist, out[:, 2:3] * (dist ** 2), out[:, 3:7] * dist], dim=-1)


# ==========================================
# 3. RZT PINN 核心类
# ==========================================
class RZT_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lx, self.Ly, self.q0, self.scale = (
            Lx,
            Ly,
            q0,
            DISPLACEMENT_SCALE,
        )
        self.net = RZT_ResNet().to(device)
        self.sobol = SobolEngine(
            dimension=2,
            scramble=True,
            seed=RANDOM_SEED,
        )
        self.adaptive_centers = None
        self.load_factor = 1.0
        self.register_buffer("props", torch.tensor(props_np))
        self.register_buffer("beta_x", torch.tensor(beta_x_np))
        self.register_buffer("beta_y", torch.tensor(beta_y_np))
        self._setup_integrators()

    def _setup_integrators(self):
        gp_np, gw_np = np.polynomial.legendre.leggauss(3)
        gp, gw = torch.tensor(gp_np), torch.tensor(gw_np)
        all_z, all_w, all_px, all_py, all_k = [], [], [], [], []
        for k in range(num_layers):
            zc, hh = (z_layers_np[k] + z_layers_np[k + 1]) / 2.0, h_k_np[k] / 2.0
            z_v = zc + gp * hh
            all_z.append(z_v)
            all_w.append(gw * hh)
            all_k.append(torch.full((3,), k, dtype=torch.long))
            pk_x_b = np.sum(beta_x_np[:k] * h_k_np[:k]) if k > 0 else 0.0
            pk_y_b = np.sum(beta_y_np[:k] * h_k_np[:k]) if k > 0 else 0.0
            all_px.append(pk_x_b + beta_x_np[k] * (z_v - z_layers_np[k]))
            all_py.append(pk_y_b + beta_y_np[k] * (z_v - z_layers_np[k]))

        self.register_buffer("Z", torch.cat(all_z).view(1, -1))
        self.register_buffer("W", torch.cat(all_w).view(1, -1))
        self.register_buffer("PX", torch.cat(all_px).view(1, -1))
        self.register_buffer("PY", torch.cat(all_py).view(1, -1))
        self.register_buffer("K", torch.cat(all_k))

        pts, wts = np.polynomial.legendre.leggauss(50)
        gx, gy = torch.meshgrid(torch.tensor(pts), torch.tensor(pts), indexing="ij")
        self.register_buffer("gauss_pts", torch.stack([gx.flatten(), gy.flatten()], dim=1))
        wt_t = torch.tensor(wts)
        self.register_buffer("gauss_wts", (wt_t.unsqueeze(0) * wt_t.unsqueeze(1)).flatten().view(-1, 1))

    def compute_energy(self, x, is_lbfgs=False, return_density=False):
        needs_graph = not return_density
        x = x.detach().requires_grad_(True)
        out = self.net(x) * self.scale
        u0, v0, w0, tx, ty, px, py = out.split(1, -1)

        def grad(f):
            return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=needs_graph, retain_graph=True)[0]

        gu, gv, gw_g, gtx, gty, gpx, gpy = map(grad, [u0, v0, w0, tx, ty, px, py])
        jx, jy = 2. / self.Lx, 2. / self.Ly
        ux, uy, vx, vy, wx, wy = gu[:, 0:1] * jx, gu[:, 1:2] * jy, gv[:, 0:1] * jx, gv[:, 1:2] * jy, gw_g[:, 0:1] * jx, gw_g[:, 1:2] * jy
        txx, txy, tyx, tyy = gtx[:, 0:1] * jx, gtx[:, 1:2] * jy, gty[:, 0:1] * jx, gty[:, 1:2] * jy
        pxx, pxy, pyx, pyy = gpx[:, 0:1] * jx, gpx[:, 1:2] * jy, gpy[:, 0:1] * jx, gpy[:, 1:2] * jy

        Ex, Ey = ux + self.Z * txx + self.PX * pxx, vy + self.Z * tyy + self.PY * pyy
        Gxy = (uy + vx) + self.Z * (txy + tyx) + self.PX * pxy + self.PY * pyx
        Gxz, Gyz = (wx + tx) + self.beta_x[self.K] * px, (wy + ty) + self.beta_y[self.K] * py

        Q = self.props[self.K]
        Um = 0.5 * (Q[:, 0] * Ex ** 2 + Q[:, 1] * Ey ** 2 + 2 * Q[:, 2] * Ex * Ey + Q[:, 3] * Gxy ** 2 +
                    2 * Q[:, 4] * Ex * Gxy + 2 * Q[:, 5] * Ey * Gxy +
                    Q[:, 6] * Gxz ** 2 + Q[:, 7] * Gyz ** 2 + 2 * Q[:, 8] * Gxz * Gyz)

        Ud = torch.sum(Um * self.W, dim=1, keepdim=True)
        if return_density: return Ud, w0

        # ==========================================
        # 载荷修改部分 (均匀载荷 -> 空间非均匀载荷)
        # ==========================================
        # 将自然坐标映射回物理坐标 x，并将单位转换为 mm
        x_m = x[:, 0:1] * (self.Lx / 2.0)
        x_mm = x_m * 1000.0

        # 根据指定公式施加载荷: 120*sin(x/10) + 120*cos(x/10) kPa
        angle_rad= (x_mm / 10.0) * (np.pi / 180.0)
        q_shape = torch.sin(angle_rad) + torch.cos(angle_rad)
        q_val = self.q0 * self.load_factor * q_shape


        area = self.Lx * self.Ly
        if is_lbfgs:
            return torch.sum(Ud * self.gauss_wts) * (area / 4.) - torch.sum(w0 * q_val * self.gauss_wts) * (area / 4.)
        return torch.mean(Ud) * area - torch.mean(w0 * q_val) * area

    def scan_adaptive_regions(self):
        sx = torch.linspace(-0.99, 0.99, 50, device=device)
        sy = torch.linspace(-0.99, 0.99, 50, device=device)
        gx, gy = torch.meshgrid(sx, sy, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        Ud, _ = self.compute_energy(pts, return_density=True)
        _, indices = torch.topk(Ud.detach().squeeze(), 400)
        self.adaptive_centers = pts[indices].detach()

    def generate_samples(self, n=2000):
        pts_uni = self.sobol.draw(n).to(device) * 2.0 - 1.0
        if self.adaptive_centers is not None:
            n_ad = int(n * 0.4)
            idx = torch.randint(
                0,
                len(self.adaptive_centers),
                (n_ad,),
                device=device,
            )
            noise = torch.randn(n_ad, 2, device=device) * 0.02
            pts_ad = (self.adaptive_centers[idx] + noise).clamp(-0.99, 0.99)
            return torch.cat([pts_uni[:n - n_ad], pts_ad], dim=0)
        return pts_uni


# ==========================================
# 4. 训练与后处理
# ==========================================
def train():
    set_seed(RANDOM_SEED)
    model = RZT_PINN().to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=0.0,
    )
    pbar = trange(8000, desc="AdamW Training")  # 优化：6000 -> 4000
    for e in pbar:
        model.load_factor = min(1.0, (e + 1) / LOAD_RAMP_EPOCHS)
        if e >= 2000: model.scan_adaptive_regions()
        pts = model.generate_samples(n=6000)
        opt.zero_grad()
        loss = model.compute_energy(pts)
        (loss / ENERGY_NORMALIZATION).backward()
        opt.step()
        if e % 200 == 0:
            pbar.set_postfix(
                {"normalized_energy": f"{loss.item() / ENERGY_NORMALIZATION:.6e}"}
            )

    print("\nL-BFGS Refinement...")
    model.load_factor = 1.0
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=LBFGS_MAX_ITER,
        history_size=50,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        l = model.compute_energy(model.gauss_pts, is_lbfgs=True)
        (l / ENERGY_NORMALIZATION).backward()
        return l / ENERGY_NORMALIZATION

    lbfgs.step(closure)
    model.zero_grad(set_to_none=True)
    diagnostic_energy = model.compute_energy(model.gauss_pts, is_lbfgs=True)
    (diagnostic_energy / ENERGY_NORMALIZATION).backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )
    model.zero_grad(set_to_none=True)
    print(
        f"Final normalized energy={diagnostic_energy.item() / ENERGY_NORMALIZATION:.8e}, "
        f"gradient norm={gradient_norm:.8e}"
    )
    return model


def postprocess(model):
    print("\n--- 执行绘图分析 ---")
    model.eval()
    cpt = torch.tensor([[0.0, 0.0]], device=device)
    with torch.no_grad():
        out_c = model.net(cpt) * model.scale
    w_center = out_c[0, 2].item() * 1000.0

    Nx, Ny = 201, 101
    xi = torch.linspace(-1, 1, Nx, device=device)
    eta = torch.linspace(-1, 1, Ny, device=device)
    XI, ETA = torch.meshgrid(xi, eta, indexing='ij')
    pts = torch.stack([XI.flatten(), ETA.flatten()], dim=1)

    with torch.no_grad():
        out = model.net(pts) * model.scale

    u0, v0, w0, tx, ty, px, py = [out[:, i].reshape(Nx, Ny).cpu().numpy() for i in range(7)]
    u_top = (u0 + (h_total / 2.0) * tx) * 1000
    v_top = (v0 + (h_total / 2.0) * ty) * 1000
    X, Y = XI.cpu().numpy() * Lx / 2, ETA.cpu().numpy() * Ly / 2

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fields = [(u_top, r'$u_{top}$ (mm)'), (v_top, r'$v_{top}$ (mm)'), (w0 * 1000, r'$w_0$ (mm)'),
              (tx, r'$\theta_x$ (rad)'), (ty, r'$\theta_y$ (rad)'), (px, r'$\psi_x$ (rad)'), (py, r'$\psi_y$ (rad)')]

    for idx, (field, title) in enumerate(fields):
        ax = axes[idx // 4, idx % 4]
        vmin, vmax = np.min(field), np.max(field)
        levels = np.linspace(vmin, vmax, 50)
        cf = ax.contourf(X, Y, field, levels=levels, cmap='jet')
        cbar = plt.colorbar(cf, ax=ax, format=FormatStrFormatter('%.3e'))
        cbar.set_ticks(np.linspace(vmin, vmax, 7))
        ax.set_title(title)
        ax.set_aspect('equal')

    ax_last = axes[1, 3]
    ax_last.plot(X[:, Ny // 2], w0[:, Ny // 2] * 1000, 'b-')
    global_w_max = float(np.max(np.abs(w0 * 1000.0)))
    ax_last.set_title(f'Centerline w (global W_max={global_w_max:.6e} mm)')
    ax_last.grid(True)
    plt.tight_layout()
    plt.show()

    # Z剖面图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    z_p = np.linspace(-h_total / 2, h_total / 2, 100)
    u_c, v_c, tx_c, ty_c, px_c, py_c = [v.item() for v in out_c[0][[0, 1, 3, 4, 5, 6]]]
    u_z, v_z = [], []
    for z in z_p:
        k = max(0, min(np.searchsorted(z_layers_np, z) - 1, num_layers - 1))
        pk_x = sum(beta_x_np[j] * h_k_np[j] for j in range(k)) + beta_x_np[k] * (z - z_layers_np[k])
        pk_y = sum(beta_y_np[j] * h_k_np[j] for j in range(k)) + beta_y_np[k] * (z - z_layers_np[k])
        u_z.append(u_c + z * tx_c + pk_x * px_c)
        v_z.append(v_c + z * ty_c + pk_y * py_c)

    axes2[0].plot(np.array(u_z) * 1000, z_p * 1000)
    axes2[0].set_title('u(z) at center (mm)')
    axes2[0].grid(True)
    axes2[1].plot(np.array(v_z) * 1000, z_p * 1000, 'r')
    axes2[1].set_title('v(z) at center (mm)')
    axes2[1].grid(True)
    axes2[2].plot([w_center] * 2, [z_p[0] * 1000, z_p[-1] * 1000], 'g--')
    axes2[2].set_title(f'w_center = {w_center:.6e} mm')
    axes2[2].grid(True)
    plt.tight_layout()
    plt.show()
    print(f'Global W_max = {global_w_max:.8e} mm')
    print(f'Center W = {w_center:.8e} mm')
    return global_w_max


def compare_deflection_across_thickness(model):
    """Display FEM w(z) and the RZT in-plane zigzag profiles."""
    print("\n--- 厚度方向 z 向位移对比 ---")
    excel_path = SCRIPT_DIRECTORY / "Deflection across thickness of case 3.xlsx"
    if not excel_path.exists():
        print(f"未找到有限元数据文件: {excel_path}")
        return

    required = {
        "X Coordinate (mm)",
        "Y Coordinate (mm)",
        "Z Coordinate (mm)",
        "Directional Deformation (mm)",
    }
    try:
        fem = pd.read_excel(excel_path)
        missing = required.difference(fem.columns)
        if missing:
            raise ValueError(f"缺少列: {sorted(missing)}")
        for column in required:
            fem[column] = pd.to_numeric(fem[column], errors="coerce")
        fem = fem.dropna(subset=list(required)).copy()
    except Exception as exc:
        print(f"读取有限元厚度位移数据失败: {exc}")
        return

    centre = fem[
        np.isclose(fem["X Coordinate (mm)"], 0.0)
        & np.isclose(fem["Y Coordinate (mm)"], 0.0)
    ].copy()
    if centre.empty:
        groups = fem.groupby(["X Coordinate (mm)", "Y Coordinate (mm)"])
        _, centre = max(groups, key=lambda item: item[1]["Z Coordinate (mm)"].nunique())

    z_half_mm = h_total * 500.0
    centre["z_model_mm"] = centre["Z Coordinate (mm)"] - z_half_mm
    centre["z_bottom_mm"] = centre["z_model_mm"] + z_half_mm
    z_min_mm, z_max_mm = z_layers_np[0] * 1000.0, z_layers_np[-1] * 1000.0
    centre = centre[
        centre["z_model_mm"].between(z_min_mm - 1.0e-9, z_max_mm + 1.0e-9)
    ].sort_values("z_model_mm")
    if centre.empty:
        print("有限元厚度位移数据不在当前模型厚度范围内。")
        return

    x_ref = float(centre["X Coordinate (mm)"].iloc[0])
    y_ref = float(centre["Y Coordinate (mm)"].iloc[0])
    xi_ref = 2.0 * (x_ref / 1000.0) / Lx
    eta_ref = 2.0 * (y_ref / 1000.0) / Ly
    if not (-1.0 <= xi_ref <= 1.0 and -1.0 <= eta_ref <= 1.0):
        print(f"有限元测点 ({x_ref:g}, {y_ref:g}) mm 超出 PINN 平面范围。")
        return

    point = torch.tensor([[xi_ref, eta_ref]], dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        output = model.net(point) * model.scale
        u_c, v_c, w_pinn_mm, tx_c, ty_c, px_c, py_c = output[0].cpu().numpy()
    w_pinn_mm *= 1000.0

    z_fem = centre["z_bottom_mm"].to_numpy(float)
    w_fem = centre["Directional Deformation (mm)"].to_numpy(float)
    error = w_pinn_mm - w_fem
    print(f"FEM profile: X={x_ref:g} mm, Y={y_ref:g} mm, points={len(centre)}")
    print(f"PINN w(z) = {w_pinn_mm:.6f} mm (RZT through-thickness constant)")
    print("说明：经典 RZT 中 w^(k)=w0，因此 w(z) 沿厚度为直线；之字形出现在 u(z) 和 v(z)。")
    print(
        f"z-displacement RMSE = {np.sqrt(np.mean(error ** 2)):.6e} mm, "
        f"MAE = {np.mean(np.abs(error)):.6e} mm"
    )

    z_plot_m = np.linspace(-h_total / 2.0, h_total / 2.0, 200)
    u_z, v_z = [], []
    for z in z_plot_m:
        layer = max(0, min(np.searchsorted(z_layers_np, z, side="right") - 1, num_layers - 1))
        phi_x = sum(beta_x_np[j] * h_k_np[j] for j in range(layer))
        phi_y = sum(beta_y_np[j] * h_k_np[j] for j in range(layer))
        phi_x += beta_x_np[layer] * (z - z_layers_np[layer])
        phi_y += beta_y_np[layer] * (z - z_layers_np[layer])
        u_z.append(u_c + z * tx_c + phi_x * px_c)
        v_z.append(v_c + z * ty_c + phi_y * py_c)

    z_bottom_plot = (z_plot_m + h_total / 2.0) * 1000.0
    interfaces = (z_layers_np[1:-1] - z_layers_np[0]) * 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5))

    axes[0].plot(
        np.full_like(z_bottom_plot, w_pinn_mm),
        z_bottom_plot,
        color="tab:blue",
        lw=2.2,
        label="RZT-PINN $w(z)$",
    )
    axes[0].scatter(
        w_fem,
        z_fem,
        s=22,
        facecolors="none",
        edgecolors="black",
        label="FEM",
    )
    axes[0].set_xlabel("z-displacement $w$ [mm]")
    axes[0].set_title("(a) $w(z)$: RZT is constant through thickness")
    axes[0].legend()

    axes[1].plot(np.asarray(u_z) * 1000.0, z_bottom_plot, color="tab:orange", lw=2.2)
    axes[1].set_xlabel("in-plane displacement $u(z)$ [mm]")
    axes[1].set_title("(b) RZT zigzag $u(z)$")

    axes[2].plot(np.asarray(v_z) * 1000.0, z_bottom_plot, color="tab:green", lw=2.2)
    axes[2].set_xlabel("in-plane displacement $v(z)$ [mm]")
    axes[2].set_title("(c) RZT zigzag $v(z)$")

    for ax in axes:
        for interface in interfaces:
            ax.axhline(interface, color="0.55", ls="--", lw=0.9)
        ax.set_ylabel("z from bottom surface [mm]")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def compare_u_across_thickness(model):
    """Display the FEM and RZT-PINN in-plane x-displacement profile u(z)."""
    print("\n--- 厚度方向 x 向位移 u(z) 对比 ---")
    excel_path = SCRIPT_DIRECTORY / "In-plane displacemnt u across thickness of case 3.xlsx"
    if not excel_path.exists():
        print(f"未找到有限元 u(z) 数据文件: {excel_path}")
        return

    required = {
        "X Coordinate (mm)",
        "Y Coordinate (mm)",
        "Z Coordinate (mm)",
        "Directional Deformation (mm)",
    }
    try:
        fem = pd.read_excel(excel_path)
        missing = required.difference(fem.columns)
        if missing:
            raise ValueError(f"缺少列: {sorted(missing)}")
        for column in required:
            fem[column] = pd.to_numeric(fem[column], errors="coerce")
        fem = fem.dropna(subset=list(required)).copy()
    except Exception as exc:
        print(f"读取有限元 u(z) 数据失败: {exc}")
        return

    centre = fem[
        np.isclose(fem["X Coordinate (mm)"], 0.0)
        & np.isclose(fem["Y Coordinate (mm)"], 0.0)
    ].copy()
    if centre.empty:
        groups = fem.groupby(["X Coordinate (mm)", "Y Coordinate (mm)"])
        _, centre = max(groups, key=lambda item: item[1]["Z Coordinate (mm)"].nunique())

    def matches_deflection_reference(profile):
        deflection_path = SCRIPT_DIRECTORY / "Deflection across thickness of case 3.xlsx"
        if not deflection_path.exists():
            return False
        try:
            deflection = pd.read_excel(deflection_path)
            required_deflection = {
                "X Coordinate (mm)",
                "Y Coordinate (mm)",
                "Z Coordinate (mm)",
                "Directional Deformation (mm)",
            }
            if not required_deflection.issubset(deflection.columns):
                return False
            deflection = deflection[
                np.isclose(deflection["X Coordinate (mm)"], x_ref)
                & np.isclose(deflection["Y Coordinate (mm)"], y_ref)
            ].copy()
            if len(deflection) != len(profile):
                return False
            left = profile.sort_values("Z Coordinate (mm)")
            right = deflection.sort_values("Z Coordinate (mm)")
            return np.allclose(
                left["Z Coordinate (mm)"].to_numpy(float),
                right["Z Coordinate (mm)"].to_numpy(float),
                rtol=0.0,
                atol=1.0e-8,
            ) and np.allclose(
                left["Directional Deformation (mm)"].to_numpy(float),
                right["Directional Deformation (mm)"].to_numpy(float),
                rtol=0.0,
                atol=1.0e-8,
            )
        except Exception:
            return False

    z_half_mm = h_total * 500.0
    centre["z_model_mm"] = centre["Z Coordinate (mm)"] - z_half_mm
    centre["z_bottom_mm"] = centre["z_model_mm"] + z_half_mm
    z_min_mm, z_max_mm = z_layers_np[0] * 1000.0, z_layers_np[-1] * 1000.0
    centre = centre[
        centre["z_model_mm"].between(z_min_mm - 1.0e-9, z_max_mm + 1.0e-9)
    ].sort_values("z_model_mm")
    if centre.empty:
        print("有限元 u(z) 数据不在当前模型厚度范围内。")
        return

    x_ref = float(centre["X Coordinate (mm)"].iloc[0])
    y_ref = float(centre["Y Coordinate (mm)"].iloc[0])
    duplicate_deflection = matches_deflection_reference(centre)
    xi_ref = 2.0 * (x_ref / 1000.0) / Lx
    eta_ref = 2.0 * (y_ref / 1000.0) / Ly
    if not (-1.0 <= xi_ref <= 1.0 and -1.0 <= eta_ref <= 1.0):
        print(f"有限元测点 ({x_ref:g}, {y_ref:g}) mm 超出 PINN 平面范围。")
        return

    point = torch.tensor([[xi_ref, eta_ref]], dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        output = model.net(point) * model.scale
        u0, _, _, theta_x, _, psi_x, _ = output[0].cpu().numpy()

    z_plot_m = np.linspace(z_layers_np[0], z_layers_np[-1], 401)
    u_pinn_m = []
    for z in z_plot_m:
        layer = max(
            0,
            min(np.searchsorted(z_layers_np, z, side="right") - 1, num_layers - 1),
        )
        phi_x = sum(beta_x_np[j] * h_k_np[j] for j in range(layer))
        phi_x += beta_x_np[layer] * (z - z_layers_np[layer])
        u_pinn_m.append(u0 + z * theta_x + phi_x * psi_x)

    z_fem = centre["z_bottom_mm"].to_numpy(float)
    u_fem_mm = centre["Directional Deformation (mm)"].to_numpy(float)
    u_pinn_mm = np.asarray(u_pinn_m) * 1000.0
    z_bottom_plot = (z_plot_m - z_layers_np[0]) * 1000.0
    z_mid_bottom_mm = z_half_mm
    u_fem_mid_mm = float(np.interp(z_mid_bottom_mm, z_fem, u_fem_mm))
    u_pinn_mid_mm = float(np.interp(z_mid_bottom_mm, z_bottom_plot, u_pinn_mm))
    u_fem_relative_mm = u_fem_mm - u_fem_mid_mm
    u_pinn_relative_mm = u_pinn_mm - u_pinn_mid_mm
    u_pinn_relative_at_fem = np.interp(z_fem, z_bottom_plot, u_pinn_relative_mm)
    errors = u_pinn_relative_at_fem - u_fem_relative_mm
    print(f"FEM profile: X={x_ref:g} mm, Y={y_ref:g} mm, points={len(centre)}")
    if duplicate_deflection:
        print(
            "警告：当前 u(z) Excel 中心点数据与 Deflection across thickness of case 3.xlsx 完全相同；"
            "请确认该文件是否误用了 w(z) 数据。"
        )
    print(
        f"采用中面刚体平移对齐：FEM u_mid={u_fem_mid_mm:.6f} mm, "
        f"PINN u_mid={u_pinn_mid_mm:.6f} mm"
    )
    print(
        f"relative u(z) error (PINN - FEM): RMSE = {np.sqrt(np.mean(errors ** 2)):.6e} mm, "
        f"MAE = {np.mean(np.abs(errors)):.6e} mm, "
        f"max|error| = {np.max(np.abs(errors)):.6e} mm"
    )

    interfaces = (z_layers_np[1:-1] - z_layers_np[0]) * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=True)
    axes[0].plot(
        u_pinn_mm, z_bottom_plot,
        color="tab:blue",
        lw=2.2,
        label="RZT-PINN",
    )
    axes[0].scatter(
        u_fem_mm, z_fem,
        s=24,
        facecolors="none",
        edgecolors="black",
        label="FEM",
    )
    axes[0].set_xlabel("absolute x-displacement $u(z)$ [mm]")
    axes[0].set_title("Absolute profile")
    axes[1].plot(
        u_pinn_relative_mm, z_bottom_plot,
        color="tab:blue",
        lw=2.2,
        label="RZT-PINN (mid-plane aligned)",
    )
    axes[1].scatter(
        u_fem_relative_mm, z_fem,
        s=24,
        facecolors="none",
        edgecolors="black",
        label="FEM (mid-plane aligned)",
    )
    axes[1].set_xlabel("relative x-displacement $u(z)-u(0)$ [mm]")
    axes[1].set_title("Profile after rigid-translation alignment")
    for ax in axes:
        for interface in interfaces:
            ax.axhline(interface, color="0.55", ls="--", lw=0.9)
        ax.set_ylabel("z from bottom surface [mm]")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Through-thickness $u(z)$ at X={x_ref:g} mm, Y={y_ref:g} mm")
    fig.tight_layout()
    plt.show()


# ==========================================
# 5. 有限元对比完整程序
# ==========================================
def compare_with_simulation(model):
    print("\n--- 执行有限元对比 ---")
    Nx, Ny = 201, 101
    xi = torch.linspace(-1, 1, Nx, device=device)
    eta = torch.linspace(-1, 1, Ny, device=device)
    XI, ETA = torch.meshgrid(xi, eta, indexing='ij')
    pts = torch.stack([XI.flatten(), ETA.flatten()], dim=1)
    X_grid, Y_grid = XI.cpu().numpy() * Lx / 2, ETA.cpu().numpy() * Ly / 2

    model.eval()
    with torch.no_grad():
        out = model.net(pts) * model.scale
        u0, v0, w0, tx, ty = [out[:, i].reshape(Nx, Ny).cpu().numpy() for i in range(5)]

    u_pinn, v_pinn, w_pinn = (u0 + (h_total / 2.0) * tx) * 1000, (v0 + (h_total / 2.0) * ty) * 1000, w0 * 1000

    def load_and_interpolate(filename, grid_x, grid_y):
        try:
            df = pd.read_excel(filename)
            data = df.values

            # 假设Excel列为 [X(mm), Y(mm), Value]
            pts_sim = np.column_stack(((data[:, 0] / 1000.0), (data[:, 1] / 1000.0)))
            vals_sim = data[:, -1] / 1000.0
            grid_val = griddata(pts_sim, vals_sim, (grid_x, grid_y), method='cubic')
            if np.isnan(grid_val).any():
                grid_val[np.isnan(grid_val)] = griddata(pts_sim, vals_sim, (grid_x, grid_y), method='nearest')[
                    np.isnan(grid_val)]
            return grid_val
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            return np.zeros_like(grid_x)

    # 加载模拟数据（需确保文件存在）
    u_sim = load_and_interpolate(
        SCRIPT_DIRECTORY / "3XSC.xlsx",
        X_grid,
        Y_grid,
    )
    v_sim = load_and_interpolate(
        SCRIPT_DIRECTORY / "3YSC.xlsx",
        X_grid,
        Y_grid,
    )
    w_sim = load_and_interpolate(
        SCRIPT_DIRECTORY / "3ZSC.xlsx",
        X_grid,
        Y_grid,
    )

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'  # 确保数学公式字体也兼容

    fig, axes = plt.subplots(3, 3, figsize=(18, 10.8), gridspec_kw={'wspace': 0.15, 'hspace': 0.25})
    plot_data = [(u_pinn, u_sim, 'u_top [mm]'), (v_pinn, v_sim, 'v_top [mm]'), (w_pinn, w_sim, 'w_0 [mm]')]
    col_labels = ["PINN Calculated", "FEM Simulated", "Error (Calc - Sim)"]

    for i, (calc, sim, title_name) in enumerate(plot_data):
        row_vals = [calc, sim, calc - sim]
        for j in range(3):
            data = row_vals[j]
            cmap = 'jet' if j < 2 else 'coolwarm'
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(vmin, vmax, 100)
            cf = axes[i, j].contourf(X_grid, Y_grid, data, levels=levels, cmap=cmap)
            # axes[i, j].set_title(f"{col_labels[j]}\n{title_name}")
            divider = make_axes_locatable(axes[i, j])
            cax = divider.append_axes("right", size="3%", pad=0.15)

            cbar = plt.colorbar(cf, cax=cax, format=FormatStrFormatter('%.3f'))
            tick_values_mm = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(tick_values_mm)
            cbar.set_ticklabels([f"{value_mm * 1000.0:.3f}" for value_mm in tick_values_mm])
            cbar.ax.tick_params(labelsize=17)
            axes[i, j].set_aspect('equal')
            axes[i, j].set_xticks([])  # 移除下侧X轴刻度
            axes[i, j].set_yticks([])  # 移除左侧Y轴刻度

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trained_model = train()
    w_max = postprocess(trained_model)
    print(f"\n最终计算最大挠度: {w_max:.4f} mm")
    compare_deflection_across_thickness(trained_model)
    compare_u_across_thickness(trained_model)
    compare_with_simulation(trained_model)
