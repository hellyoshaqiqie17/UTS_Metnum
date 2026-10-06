"""
================================================================================
UJIAN TENGAH SEMESTER GASAL 2026/2027 - METODE NUMERIK (FTM25602016)
BAGIAN 2: CODING (TAKE HOME) - SOAL 5 (BERBASIS DATA SOAL 4)
Program Studi: Teknik Robotika dan Kecerdasan Buatan, FTMM Universitas Airlangga
Mahasiswa    : Hellyos Ageng Haqiqie (NIM: 163251001)
================================================================================
Deskripsi:
Simulasi numerik pemodelan regresi linier metode Least Squares secara mandiri
(from scratch) tanpa fungsi regresi bawaan (built-in).
Mengevaluasi model menggunakan residual, SSE, MSE, RMSE, dan R^2.
Mengestimasi input kendali motor x untuk target output y = 2.75 m/s.
Membandingkan solusi analitik vs numerik serta interpretasi fisis robotika.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Konfigurasi style plotting agar elegan, modern, dan berstandar publikasi ilmiah
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

def manual_least_squares(x, y):
    """
    Menghitung parameter regresi linier y_hat = a0 + a1 * x
    menggunakan formulasi analitik Least Squares (Normal Equations) secara manual.
    
    Rumus:
    a1 = (m * sum(xy) - sum(x) * sum(y)) / (m * sum(x^2) - (sum(x))^2)
    a0 = mean(y) - a1 * mean(x)
    """
    m = len(x)
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    
    for xi, yi in zip(x, y):
        sum_x += xi
        sum_y += yi
        sum_xx += xi * xi
        sum_xy += xi * yi
        
    x_bar = sum_x / m
    y_bar = sum_y / m
    
    # Menghitung koefisien a1 (kemiringan / slope)
    pembilang_a1 = m * sum_xy - sum_x * sum_y
    penyebut_a1 = m * sum_xx - (sum_x ** 2)
    a1 = pembilang_a1 / penyebut_a1
    
    # Menghitung koefisien a0 (intersep / titik potong)
    a0 = y_bar - a1 * x_bar
    
    return a0, a1, {
        'm': m,
        'sum_x': sum_x,
        'sum_y': sum_y,
        'sum_xx': sum_xx,
        'sum_xy': sum_xy,
        'x_bar': x_bar,
        'y_bar': y_bar,
        'denom': penyebut_a1
    }

def manual_matrix_least_squares(x, y):
    """
    Menyelesaikan Sistem Persamaan Normal (X^T X) a = X^T y
    menggunakan Eliminasi Gauss manual.
    """
    m = len(x)
    # Matriks desain X [m x 2]: kolom 1 adalah 1, kolom 2 adalah x
    X = np.column_stack([np.ones(m), x])
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Eliminasi Gauss 2x2 manual
    # [ [A00, A01 | B0],
    #   [A10, A11 | B1] ]
    A = XTX.copy().astype(float)
    B = XTy.copy().astype(float)
    
    factor = A[1, 0] / A[0, 0]
    A[1, 1] -= factor * A[0, 1]
    B[1] -= factor * B[0]
    
    a1_num = B[1] / A[1, 1]
    a0_num = (B[0] - A[0, 1] * a1_num) / A[0, 0]
    
    return a0_num, a1_num

def evaluate_model(x, y, a0, a1):
    """
    Menghitung prediksi, residual, dan metrik performa model:
    SSE, MSE, RMSE, MAE, dan R^2.
    """
    m = len(x)
    y_pred = a0 + a1 * x
    residuals = y - y_pred
    
    sse = np.sum(residuals ** 2)
    mse = sse / m
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    y_bar = np.mean(y)
    sst = np.sum((y - y_bar) ** 2)
    r2 = 1.0 - (sse / sst)
    
    return {
        'y_pred': y_pred,
        'residuals': residuals,
        'sse': sse,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'sst': sst,
        'r2': r2
    }

def estimate_input_for_target(a0, a1, y_target):
    """
    Mengestimasi nilai input motor x ketika target output y_target diketahui.
    Metode analitik: x = (y_target - a0) / a1
    Metode numerik: Root-finding Newton-Raphson untuk f(x) = a0 + a1*x - y_target = 0
    """
    # 1. Solusi Analitik
    x_analitik = (y_target - a0) / a1
    
    # 2. Solusi Numerik (Newton-Raphson)
    # f(x) = (a0 + a1*x) - y_target
    # f'(x) = a1
    x_curr = 1.0  # Tebakan awal
    tol = 1e-12
    max_iter = 50
    for iter_idx in range(max_iter):
        f_val = (a0 + a1 * x_curr) - y_target
        f_prime = a1
        x_next = x_curr - f_val / f_prime
        if abs(x_next - x_curr) < tol:
            break
        x_curr = x_next
    x_numerik = x_next
    
    return x_analitik, x_numerik

def run_simulation(output_dir="."):
    """
    Menjalankan seluruh tahapan perhitungan, visualisasi, dan pelaporan Soal 5.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Data pengujian dari Soal 4
    # x : Input kendali motor
    # y : Kecepatan aktual mobile robot (m/s)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    y = np.array([1.2, 1.9, 3.2, 3.8, 5.1], dtype=float)
    
    print("=" * 75)
    print("ANALISIS REGRESI LINIER LEAST SQUARES (SOAL 5 & 4)")
    print("=" * 75)
    print(f"Data Pengujian Robotika:")
    for xi, yi in zip(x, y):
        print(f"  Input x = {xi:4.1f}  -->  Kecepatan aktual y = {yi:4.2f} m/s")
    print("-" * 75)
    
    # Perhitungan Least Squares Manual
    a0, a1, stats = manual_least_squares(x, y)
    a0_mat, a1_mat = manual_matrix_least_squares(x, y)
    
    print("\n[1] HASIL PERHITUNGAN PARAMETER MODEL (y_hat = a0 + a1 * x):")
    print(f"  Jumlah data (m)        : {stats['m']}")
    print(f"  Sigma x               : {stats['sum_x']:.4f}")
    print(f"  Sigma y               : {stats['sum_y']:.4f}")
    print(f"  Sigma x^2             : {stats['sum_xx']:.4f}")
    print(f"  Sigma x*y             : {stats['sum_xy']:.4f}")
    print(f"  Rata-rata x (x_bar)   : {stats['x_bar']:.4f}")
    print(f"  Rata-rata y (y_bar)   : {stats['y_bar']:.4f}")
    print(f"  Penyebut (D)          : {stats['denom']:.4f}")
    print(f"  Intersep (a0)         : {a0:.6f}  (Matriks Gauss: {a0_mat:.6f})")
    print(f"  Kemiringan (a1)       : {a1:.6f}  (Matriks Gauss: {a1_mat:.6f})")
    print(f"  Persamaan Model       : y_hat = {a0:.4f} + {a1:.4f} * x")
    
    # Evaluasi Model
    eval_res = evaluate_model(x, y, a0, a1)
    print("\n[2] TABEL EVALUASI DATA, PREDIKSI, DAN RESIDUAL:")
    print("  +---+-------+-------+---------+----------+------------+")
    print("  | i |   x   |   y   |  y_hat  | Residual | Residual^2 |")
    print("  +---+-------+-------+---------+----------+------------+")
    for i in range(len(x)):
        print(f"  | {i+1} | {x[i]:5.1f} | {y[i]:5.2f} | {eval_res['y_pred'][i]:7.4f} | "
              f"{eval_res['residuals'][i]:8.4f} | {eval_res['residuals'][i]**2:10.6f} |")
    print("  +---+-------+-------+---------+----------+------------+")
    print(f"  Sum of Residuals      : {np.sum(eval_res['residuals']):.2e} (Mendekati 0 secara teoretis)")
    print(f"  Sum of Squared Errors : {eval_res['sse']:.6f}")
    print(f"  Mean Squared Error    : {eval_res['mse']:.6f}")
    print(f"  Root Mean Squared Error (RMSE) : {eval_res['rmse']:.6f} m/s")
    print(f"  Mean Absolute Error (MAE)      : {eval_res['mae']:.6f} m/s")
    print(f"  Koefisien Determinasi (R^2)   : {eval_res['r2']:.6f} ({eval_res['r2']*100:.2f}%)")
    
    # Estimasi Input untuk Output y = 2.75 m/s
    y_target = 2.75
    x_analitik, x_numerik = estimate_input_for_target(a0, a1, y_target)
    print(f"\n[3] ESTIMASI NILAI INPUT UNTUK OUTPUT y = {y_target} m/s:")
    print(f"  Target Output (y)     : {y_target} m/s")
    print(f"  Hasil Analitik (x)    : {x_analitik:.6f}  (Eksak: (2.75 - 0.13)/0.97 = 262/97)")
    print(f"  Hasil Numerik (x)     : {x_numerik:.6f}  (Metode Newton-Raphson)")
    print(f"  Selisih (|Analitik - Numerik|) : {abs(x_analitik - x_numerik):.2e}")
    
    # Verifikasi dengan built-in NumPy (hanya untuk cross-check independen)
    poly_coeff = np.polyfit(x, y, 1) # polyfit returns [slope, intercept]
    print(f"\n[4] VALIDASI INDEPENDEN (NUMPY BUILT-IN):")
    print(f"  np.polyfit Slope      : {poly_coeff[0]:.6f} (Selisih: {abs(a1 - poly_coeff[0]):.2e})")
    print(f"  np.polyfit Intercept  : {poly_coeff[1]:.6f} (Selisih: {abs(a0 - poly_coeff[1]):.2e})")
    
    # -------------------------------------------------------------
    # Visualisasi 1: Kurva Regresi Linier dan Titik Estimasi Target
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    x_fine = np.linspace(0.5, 5.5, 200)
    y_fine = a0 + a1 * x_fine
    
    # Garis regresi
    ax.plot(x_fine, y_fine, color='#1f77b4', linestyle='-', linewidth=2.2, 
            label=f'Model Regresi: $\\hat{{y}} = {a0:.2f} + {a1:.2f}x$')
    
    # Titik data aktual
    ax.scatter(x, y, color='#d62728', s=80, zorder=5, edgecolors='black', linewidth=1.2,
               label='Data Pengukuran Motor')
    
    # Segmen residual (jarak vertikal dari titik data ke garis)
    for xi, yi, ypi in zip(x, y, eval_res['y_pred']):
        ax.plot([xi, xi], [yi, ypi], color='#7f7f7f', linestyle=':', linewidth=1.5, zorder=4)
    # Dummy plot untuk legend residual
    ax.plot([], [], color='#7f7f7f', linestyle=':', label='Residual ($e_i = y_i - \\hat{y}_i$)')
    
    # Titik estimasi input kendali untuk target output y = 2.75
    ax.scatter([x_analitik], [y_target], color='#2ca02c', s=120, marker='*', zorder=6,
               edgecolors='black', linewidth=1.2,
               label=f'Estimasi Titik Target:\n$x^* = {x_analitik:.3f}$, $y = {y_target}$ m/s')
    
    # Garis bantu proyeksi titik target
    ax.axhline(y_target, color='#2ca02c', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axvline(x_analitik, color='#2ca02c', linestyle='--', alpha=0.6, linewidth=1.2)
    
    ax.set_title('Regresi Linier Least Squares: Input Motor vs Kecepatan Robot', fontweight='bold', pad=12)
    ax.set_xlabel('Input Kendali Motor ($x$) [Unit / Skala Kendali]')
    ax.set_ylabel('Kecepatan Aktual Mobile Robot ($y$) [m/s]')
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Kotak anotasi parameter performa
    info_text = (f"$R^2 = {eval_res['r2']:.4f}$\n"
                 f"$\\mathrm{{RMSE}} = {eval_res['rmse']:.4f}$ m/s\n"
                 f"$\\mathrm{{SSE}} = {eval_res['sse']:.4f}$")
    ax.text(0.95, 0.08, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9))
    
    fig.tight_layout()
    fig1_path = os.path.join(output_dir, "fig_soal5_regression.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\n[+] Visualisasi regresi disimpan: {fig1_path}")
    
    # -------------------------------------------------------------
    # Visualisasi 2: Analisis Residual
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Plot residual vs input x
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.stem(x, eval_res['residuals'], linefmt='#1f77b4', markerfmt='o', basefmt=" ")
    ax1.set_title('Distribusi Residual terhadap Input Kendali $x$', fontweight='bold')
    ax1.set_xlabel('Input Kendali Motor ($x$)')
    ax1.set_ylabel('Residual ($e_i = y_i - \\hat{y}_i$) [m/s]')
    ax1.set_ylim(-0.35, 0.35)
    ax1.grid(True)
    
    # Plot residual vs predicted value y_hat
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.scatter(eval_res['y_pred'], eval_res['residuals'], color='#d62728', s=70, edgecolors='black', zorder=5)
    ax2.set_title('Residual terhadap Nilai Prediksi $\\hat{y}$', fontweight='bold')
    ax2.set_xlabel('Kecepatan Prediksi $\\hat{y}$ [m/s]')
    ax2.set_ylabel('Residual [m/s]')
    ax2.set_ylim(-0.35, 0.35)
    ax2.grid(True)
    
    fig.tight_layout()
    fig2_path = os.path.join(output_dir, "fig_soal5_residuals.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[+] Visualisasi residual disimpan: {fig2_path}")
    
    return {
        'a0': a0,
        'a1': a1,
        'eval': eval_res,
        'x_target': x_analitik,
        'y_target': y_target
    }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_simulation(current_dir)
