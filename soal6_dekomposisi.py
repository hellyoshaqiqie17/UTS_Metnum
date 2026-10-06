"""
================================================================================
UJIAN TENGAH SEMESTER GASAL 2026/2027 - METODE NUMERIK (FTM25602016)
BAGIAN 2: CODING (TAKE HOME) - SOAL 6 (BERBASIS FENOMENA SOAL 3)
Program Studi: Teknik Robotika dan Kecerdasan Buatan, FTMM Universitas Airlangga
Mahasiswa    : Hellyos Ageng Haqiqie (NIM: 163251001)
================================================================================
Deskripsi:
Simulasi numerik pemodelan polinomial kecepatan drone quadcopter menggunakan
Dekomposisi QR dan Dekomposisi LU mandiri (from scratch).
Parameter NIM: N = 1 (dua digit terakhir NIM 163251001).
Membandingkan:
a. Hasil analitik vs numerik (solusi x dan estimasi v(4.2 s)).
b. Keakuratan numerik (residual norm, galat absolut/relatif) dan FLOPs
   (Floating Point Operations) antara Dekomposisi QR dan Dekomposisi LU.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Konfigurasi style plotting standar publikasi ilmiah
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

class OperationCounter:
    """Kelas pelacak jumlah operasi titik-kambang (Floating-Point Operations / FLOPs)."""
    def __init__(self):
        self.adds = 0
        self.subs = 0
        self.muls = 0
        self.divs = 0
        self.sqrts = 0

    def add(self, count=1): self.adds += count
    def sub(self, count=1): self.subs += count
    def mul(self, count=1): self.muls += count
    def div(self, count=1): self.divs += count
    def sqrt(self, count=1): self.sqrts += count

    @property
    def total_flops(self):
        return self.adds + self.subs + self.muls + self.divs + self.sqrts

    def summary(self):
        return (f"Total FLOPs: {self.total_flops} (Add: {self.adds}, Sub: {self.subs}, "
                f"Mul: {self.muls}, Div: {self.divs}, Sqrt: {self.sqrts})")


# --------------------------------------------------------------------------
# 1. IMPLEMENTASI DEKOMPOSISI QR (MODIFIED GRAM-SCHMIDT) FROM SCRATCH
# --------------------------------------------------------------------------
def qr_decomposition_mgs(A):
    """
    Faktorisasi A = Q * R menggunakan Modified Gram-Schmidt (MGS).
    Q: matriks ortonormal (Q^T Q = I)
    R: matriks segitiga atas (upper triangular)
    """
    counter = OperationCounter()
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    V = A.copy().astype(np.float64)

    for j in range(n):
        # Hitung norm kolom j
        norm_v2 = 0.0
        for i in range(m):
            norm_v2 += V[i, j] * V[i, j]
            counter.mul()
            if i > 0:
                counter.add()
        R[j, j] = np.sqrt(norm_v2)
        counter.sqrt()

        # Normalisasi untuk memperoleh vektor basis Q[:, j]
        for i in range(m):
            Q[i, j] = V[i, j] / R[j, j]
            counter.div()

        # Proyeksi ortogonal pada kolom tersisa (Modified Gram-Schmidt)
        for k in range(j + 1, n):
            dot_val = 0.0
            for i in range(m):
                dot_val += Q[i, j] * V[i, k]
                counter.mul()
                if i > 0:
                    counter.add()
            R[j, k] = dot_val

            for i in range(m):
                V[i, k] -= R[j, k] * Q[i, j]
                counter.mul()
                counter.sub()

    return Q, R, counter

def solve_qr(A, b):
    """
    Menyelesaikan Ax = b menggunakan dekomposisi QR:
    1. A = Q * R
    2. R * x = Q^T * b  (karena Q^T Q = I)
    3. Back-substitution pada sistem segitiga atas R
    """
    Q, R, qr_counter = qr_decomposition_mgs(A)
    m, n = A.shape
    counter = OperationCounter()
    counter.adds = qr_counter.adds
    counter.subs = qr_counter.subs
    counter.muls = qr_counter.muls
    counter.divs = qr_counter.divs
    counter.sqrts = qr_counter.sqrts

    # Hitung d = Q^T * b
    d = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = 0.0
        for j in range(m):
            val += Q[j, i] * b[j]
            counter.mul()
            if j > 0:
                counter.add()
        d[i] = val

    # Back-substitution: R x = d
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += R[i, j] * x[j]
            counter.mul()
            if j > i + 1:
                counter.add()
        if i + 1 < n:
            counter.sub()
        x[i] = (d[i] - s) / R[i, i]
        counter.div()

    return x, Q, R, counter


# --------------------------------------------------------------------------
# 2. IMPLEMENTASI DEKOMPOSISI LU (DOOLITTLE) FROM SCRATCH
# --------------------------------------------------------------------------
def lu_decomposition_doolittle(A):
    """
    Faktorisasi A = L * U menggunakan metode Doolittle.
    L: unit lower triangular (l_ii = 1.0)
    U: upper triangular
    """
    counter = OperationCounter()
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Baris ke-i dari U
        for k in range(i, n):
            val = A[i, k]
            for j in range(i):
                val -= L[i, j] * U[j, k]
                counter.mul()
                counter.sub()
            U[i, k] = val

        # Kolom ke-i dari L
        for k in range(i + 1, n):
            val = A[k, i]
            for j in range(i):
                val -= L[k, j] * U[j, i]
                counter.mul()
                counter.sub()
            L[k, i] = val / U[i, i]
            counter.div()

    return L, U, counter

def solve_lu(A, b):
    """
    Menyelesaikan Ax = b menggunakan dekomposisi LU:
    1. A = L * U
    2. L * y = b  (Forward substitution)
    3. U * x = y  (Back substitution)
    """
    L, U, lu_counter = lu_decomposition_doolittle(A)
    n = A.shape[0]
    counter = OperationCounter()
    counter.adds = lu_counter.adds
    counter.subs = lu_counter.subs
    counter.muls = lu_counter.muls
    counter.divs = lu_counter.divs
    counter.sqrts = lu_counter.sqrts

    # 1. Forward substitution: L * y = b (karena L_ii = 1)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
            counter.mul()
            if j > 0:
                counter.add()
        if i > 0:
            counter.sub()
        y[i] = b[i] - s

    # 2. Back substitution: U * x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
            counter.mul()
            if j > i + 1:
                counter.add()
        if i + 1 < n:
            counter.sub()
        x[i] = (y[i] - s) / U[i, i]
        counter.div()

    return x, L, U, counter


# --------------------------------------------------------------------------
# 3. SOLUSI ANALITIK EKSAK (BERDASARKAN INVERS MATRIKS VANDERMONDE)
# --------------------------------------------------------------------------
def solve_analytical(N):
    """
    Menghitung vektor kecepatan v dan solusi analitik eksak x = A^(-1) * v
    sebagai fungsi dari parameter NIM N.
    """
    # Titik data kecepatan IMU
    v1 = 2.5 + 0.02 * N
    v2 = 5.5 + 0.03 * N
    v3 = 7.0 + 0.035 * N
    v = np.array([v1, v2, v3], dtype=np.float64)

    # Invers analitik A^-1 diturunkan dari matriks kofaktor dan det(A) = -16:
    # A = [ [1, 1, 1], [9, 3, 1], [25, 5, 1] ]
    # A^-1 = [ [ 1/8,  -1/4,  1/8 ],
    #          [  -1,   3/2, -1/2 ],
    #          [15/8,  -5/4,  3/8 ] ]
    A_inv = np.array([
        [ 0.125, -0.250,  0.125],
        [-1.000,  1.500, -0.500],
        [ 1.875, -1.250,  0.375]
    ], dtype=np.float64)

    x_exact = A_inv @ v

    # Nilai analitik langsung dari penurunan aljabar:
    # x1 = -0.1875 - 0.000625 * N
    # x2 =  2.2500 + 0.007500 * N
    # x3 =  0.4375 + 0.013125 * N
    x1_form = -0.1875 - 0.000625 * N
    x2_form =  2.2500 + 0.007500 * N
    x3_form =  0.4375 + 0.013125 * N
    x_form = np.array([x1_form, x2_form, x3_form], dtype=np.float64)

    return v, x_exact, x_form, A_inv


# --------------------------------------------------------------------------
# 4. RUN SIMULASI & BENCHMARK
# --------------------------------------------------------------------------
def run_simulation(N=1, output_dir="."):
    """
    Menjalankan seluruh perbandingan Soal 6 (QR vs LU vs Analitik).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"ANALISIS DEKOMPOSISI QR VS LU PADA MODEL DRONE QUADCOPTER (SOAL 6 & 3)")
    print("=" * 80)
    print(f"Parameter NIM: N = {N} (Hellyos Ageng Haqiqie - 163251001)")

    # Waktu pengukuran dan pembentukan matriks A
    t_meas = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    A = np.array([
        [t_meas[0]**2, t_meas[0], 1.0],
        [t_meas[1]**2, t_meas[1], 1.0],
        [t_meas[2]**2, t_meas[2], 1.0]
    ], dtype=np.float64)

    v, x_exact, x_form, A_inv = solve_analytical(N)

    print(f"\n[1] DATA PENGUKURAN KECEPATAN IMU:")
    print(f"  t1 = 1.0 s  -->  v1 = 2.5 + 0.02 * {N}  = {v[0]:.4f} m/s")
    print(f"  t2 = 3.0 s  -->  v2 = 5.5 + 0.03 * {N}  = {v[1]:.4f} m/s")
    print(f"  t3 = 5.0 s  -->  v3 = 7.0 + 0.035 * {N} = {v[2]:.4f} m/s")

    print(f"\n[2] SISTEM PERSAMAAN LINIER A * x = v:")
    print("  Matriks Vandermonde A:")
    for row in A:
        print(f"    [ {row[0]:4.1f}  {row[1]:4.1f}  {row[2]:4.1f} ]")
    print("  Vektor Target v:")
    print(f"    [ {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f} ]^T")

    # Analisis Kondisi Matriks
    cond_2 = np.linalg.cond(A, 2)
    cond_inf = np.linalg.cond(A, np.inf)
    det_A = np.linalg.det(A)
    print(f"\n  Determinan A          : {det_A:.4f} (Eksak: -16.0)")
    print(f"  Condition Number L2   : {cond_2:.4f}")
    print(f"  Condition Number Linf : {cond_inf:.4f}")

    # Solusi Analitik
    t_target = 4.2
    v_target_exact = x_exact[0] * (t_target**2) + x_exact[1] * t_target + x_exact[2]

    print(f"\n[3] SOLUSI ANALITIK EKSAK:")
    print(f"  x1 (koefisien t^2)    : {x_exact[0]:.8f}  (Eksak: -13/64 = -0.203125)")
    print(f"  x2 (koefisien t)      : {x_exact[1]:.8f}  (Eksak:  39/16 =  2.437500)")
    print(f"  x3 (konstanta)        : {x_exact[2]:.8f}  (Eksak:  49/64 =  0.765625)")
    print(f"  Estimasi v({t_target} s)   : {v_target_exact:.8f} m/s")

    # -------------------------------------------------------------
    # Solusi Numerik Menggunakan Dekomposisi QR
    # -------------------------------------------------------------
    t0_qr = time.perf_counter()
    x_qr, Q, R, counter_qr = solve_qr(A, v)
    t1_qr = time.perf_counter()
    time_qr = (t1_qr - t0_qr) * 1e6  # microsecond
    v_target_qr = x_qr[0] * (t_target**2) + x_qr[1] * t_target + x_qr[2]

    # -------------------------------------------------------------
    # Solusi Numerik Menggunakan Dekomposisi LU
    # -------------------------------------------------------------
    t0_lu = time.perf_counter()
    x_lu, L, U, counter_lu = solve_lu(A, v)
    t1_lu = time.perf_counter()
    time_lu = (t1_lu - t0_lu) * 1e6  # microsecond
    v_target_lu = x_lu[0] * (t_target**2) + x_lu[1] * t_target + x_lu[2]

    # -------------------------------------------------------------
    # Evaluasi Galat dan Residual
    # -------------------------------------------------------------
    res_qr = np.linalg.norm(A @ x_qr - v, 2)
    res_lu = np.linalg.norm(A @ x_lu - v, 2)

    err_x_qr = np.linalg.norm(x_qr - x_exact, 2)
    err_x_lu = np.linalg.norm(x_lu - x_exact, 2)

    err_rel_qr = err_x_qr / np.linalg.norm(x_exact, 2)
    err_rel_lu = err_x_lu / np.linalg.norm(x_exact, 2)

    err_v_qr = abs(v_target_qr - v_target_exact)
    err_v_lu = abs(v_target_lu - v_target_exact)

    print("\n" + "=" * 80)
    print("[4] PERBANDINGAN HASIL ANALITIK VS NUMERIK (POIN A)")
    print("=" * 80)
    header = f"{'Parameter':<18} | {'Analitik Eksak':<16} | {'Dekomposisi QR':<16} | {'Dekomposisi LU':<16}"
    print(header)
    print("-" * len(header))
    print(f"{'x1 (t^2)':<18} | {x_exact[0]:<16.8f} | {x_qr[0]:<16.8f} | {x_lu[0]:<16.8f}")
    print(f"{'x2 (t)':<18} | {x_exact[1]:<16.8f} | {x_qr[1]:<16.8f} | {x_lu[1]:<16.8f}")
    print(f"{'x3 (konstanta)':<18} | {x_exact[2]:<16.8f} | {x_qr[2]:<16.8f} | {x_lu[2]:<16.8f}")
    print(f"{'v(t = 4.2 s)':<18} | {v_target_exact:<16.8f} | {v_target_qr:<16.8f} | {v_target_lu:<16.8f}")
    print("-" * len(header))

    print("\n" + "=" * 80)
    print("[5] PERBANDINGAN KEAKURATAN & FLOPs (POIN B)")
    print("=" * 80)
    header_b = f"{'Metrik Evaluasi':<28} | {'Dekomposisi QR':<22} | {'Dekomposisi LU':<22}"
    print(header_b)
    print("-" * len(header_b))
    print(f"{'Norm Residual ||Ax - v||_2':<28} | {res_qr:<22.3e} | {res_lu:<22.3e}")
    print(f"{'Galat Vektor ||x - x*||_2':<28} | {err_x_qr:<22.3e} | {err_x_lu:<22.3e}")
    print(f"{'Galat Relatif Solusi':<28} | {err_rel_qr:<22.3e} | {err_rel_lu:<22.3e}")
    print(f"{'Galat Estimasi v(4.2 s)':<28} | {err_v_qr:<22.3e} | {err_v_lu:<22.3e}")
    print(f"{'Total FLOPs Operasi':<28} | {counter_qr.total_flops:<22} | {counter_lu.total_flops:<22}")
    print(f"{'Rasio FLOPs (QR / LU)':<28} | {counter_qr.total_flops / counter_lu.total_flops:<22.2f} | {'1.00 (Basis Acuan)':<22}")
    print("-" * len(header_b))

    print(f"\nRincian Operasi FLOPs:")
    print(f"  QR Decomposition : {counter_qr.summary()}")
    print(f"  LU Decomposition : {counter_lu.summary()}")

    # -------------------------------------------------------------
    # Visualisasi 1: Kurva Kecepatan Drone v(t) & Titik Estimasi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    t_fine = np.linspace(1.0, 5.0, 300)
    v_fine_exact = x_exact[0] * (t_fine**2) + x_exact[1] * t_fine + x_exact[2]
    v_fine_qr = x_qr[0] * (t_fine**2) + x_qr[1] * t_fine + x_qr[2]
    v_fine_lu = x_lu[0] * (t_fine**2) + x_lu[1] * t_fine + x_lu[2]

    # Kurva analitik dan numerik
    ax.plot(t_fine, v_fine_exact, color='#1f77b4', linestyle='-', linewidth=2.5,
            label=f'Analitik Eksak: $v(t) = {x_exact[0]:.4f}t^2 + {x_exact[1]:.4f}t + {x_exact[2]:.4f}$')
    ax.plot(t_fine, v_fine_qr, color='#ff7f0e', linestyle='--', linewidth=1.8,
            label='Dekomposisi QR (Modified Gram-Schmidt)')
    ax.plot(t_fine, v_fine_lu, color='#2ca02c', linestyle=':', linewidth=1.8,
            label='Dekomposisi LU (Doolittle)')

    # Titik pengukuran sensor IMU
    ax.scatter(t_meas, v, color='#d62728', s=90, zorder=6, edgecolors='black', linewidth=1.2,
               label=f'Pengukuran Sensor IMU ($N={N}$)')
    for i, (ti, vi) in enumerate(zip(t_meas, v)):
        ax.annotate(f'Data {i+1}: ({ti:.1f} s, {vi:.3f} m/s)', (ti, vi),
                    textcoords="offset points", xytext=(-15, 12),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.85),
                    fontsize=8.5)

    # Titik estimasi kecepatan pada t = 4.2 s
    ax.scatter([t_target], [v_target_exact], color='#9467bd', marker='*', s=150, zorder=7,
               edgecolors='black', linewidth=1.2,
               label=f'Estimasi $t=4.2$ s: $v = {v_target_exact:.4f}$ m/s')
    ax.axvline(t_target, color='#9467bd', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axhline(v_target_exact, color='#9467bd', linestyle='--', alpha=0.6, linewidth=1.2)

    ax.set_title('Profil Kecepatan Vertikal Drone Selama Fase Thrusting', fontweight='bold', pad=12)
    ax.set_xlabel('Waktu Penerbangan $t$ [detik]')
    ax.set_ylabel('Kecepatan Vertikal $v(t)$ [m/s]')
    ax.set_xlim(0.8, 5.2)
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, framealpha=0.92)

    # Info box formula model
    model_box = (f"Model: $v(t) = x_1 t^2 + x_2 t + x_3$\n"
                 f"$x_1 = {x_exact[0]:.6f}$ m/s$^3$\n"
                 f"$x_2 = {x_exact[1]:.6f}$ m/s$^2$\n"
                 f"$x_3 = {x_exact[2]:.6f}$ m/s\n"
                 f"$v(4.2) = {v_target_exact:.4f}$ m/s")
    ax.text(0.04, 0.95, model_box, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9))

    fig.tight_layout()
    fig1_path = os.path.join(output_dir, "fig_soal6_trajectory.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\n[+] Visualisasi lintasan kecepatan drone disimpan: {fig1_path}")

    # -------------------------------------------------------------
    # Visualisasi 2: Diagram Metrik Komparasi QR vs LU
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    methods = ['Dekomposisi LU', 'Dekomposisi QR']
    colors = ['#2ca02c', '#ff7f0e']

    # Subplot 1: Total Operasi FLOPs
    flops_values = [counter_lu.total_flops, counter_qr.total_flops]
    bars1 = ax1.bar(methods, flops_values, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    ax1.set_title('Kompleksitas Komputasi (Total FLOPs)', fontweight='bold')
    ax1.set_ylabel('Jumlah FLOPs Operasi')
    ax1.set_ylim(0, max(flops_values) * 1.25)
    for bar, val in zip(bars1, flops_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val} FLOPs',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Norm Residual (Skala Logaritmik)
    residuals = [res_lu, res_qr]
    bars2 = ax2.bar(methods, residuals, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    ax2.set_yscale('log')
    ax2.set_title('Keakuratan Numerik (Norm Residual $||Ax - v||_2$)', fontweight='bold')
    ax2.set_ylabel('Residual Norm (Skala Logaritmik)')
    ax2.set_ylim(1e-16, 1e-12)
    for bar, val in zip(bars2, residuals):
        ax2.text(bar.get_x() + bar.get_width()/2, val * 1.5, f'{val:.2e}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig2_path = os.path.join(output_dir, "fig_soal6_metrics.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[+] Visualisasi perbandingan metrik QR vs LU disimpan: {fig2_path}")

    return {
        'x_exact': x_exact,
        'x_qr': x_qr,
        'x_lu': x_lu,
        'v_target_exact': v_target_exact,
        'v_target_qr': v_target_qr,
        'v_target_lu': v_target_lu,
        'res_qr': res_qr,
        'res_lu': res_lu,
        'counter_qr': counter_qr,
        'counter_lu': counter_lu,
        'cond_2': cond_2
    }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_simulation(N=1, output_dir=current_dir)
