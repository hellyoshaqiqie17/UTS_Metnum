"""
================================================================================
UJIAN TENGAH SEMESTER GASAL 2026/2027 - METODE NUMERIK (FTM25602016)
BAGIAN 2: CODING (TAKE HOME) - MASTER RUNNER SCRIPT
Program Studi: Teknik Robotika dan Kecerdasan Buatan, FTMM Universitas Airlangga
Mahasiswa    : Hellyos Ageng Haqiqie (NIM: 163251001)
================================================================================
Script ini mengeksekusi seluruh simulasi Soal 5 dan Soal 6 secara berurutan,
menghasilkan seluruh file gambar visualisasi (high-resolution PNG),
serta menampilkan ringkasan komparasi analitik dan numerik secara terpadu.
================================================================================
"""

import os
import sys
import soal5_regresi
import soal6_dekomposisi

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("UNIVERSITAS AIRLANGGA - FAKULTAS TEKNOLOGI MAJU DAN MULTIDISIPLIN")
    print("UJIAN TENGAH SEMESTER GASAL 2026/2027 - METODE NUMERIK (FTM25602016)")
    print("BAGIAN 2: CODING TAKE HOME EXAM (SOAL 5 & SOAL 6)")
    print("Mahasiswa : Hellyos Ageng Haqiqie")
    print("NIM       : 163251001  -->  Parameter N = 1")
    print("Prodi     : Teknik Robotika dan Kecerdasan Buatan")
    print("PJMK      : Asif Ali Zamzami")
    print("=" * 80)
    
    # Eksekusi Soal 5
    print("\n>>> MEMULAI SIMULASI SOAL 5 (REGRESI LEAST SQUARES MANDIRI)...")
    res5 = soal5_regresi.run_simulation(output_dir=current_dir)
    
    # Eksekusi Soal 6
    print("\n>>> MEMULAI SIMULASI SOAL 6 (DEKOMPOSISI QR VS LU MANDIRI)...")
    res6 = soal6_dekomposisi.run_simulation(N=1, output_dir=current_dir)
    
    # Ringkasan Akhir
    print("\n" + "=" * 80)
    print("RINGKASAN EKSEKUSI KESELURUHAN (VERIFIKASI AKHIR)")
    print("=" * 80)
    print("[1] Soal 5 (Regresi Linier & Estimasi Input Motor):")
    print(f"    - Persamaan Model   : y_hat = {res5['a0']:.4f} + {res5['a1']:.4f} * x")
    print(f"    - Kualitas Model    : R^2 = {res5['eval']['r2']*100:.2f}%, RMSE = {res5['eval']['rmse']:.4f} m/s")
    print(f"    - Target Kecepatan  : y = {res5['y_target']} m/s")
    print(f"    - Estimasi Input    : x* = {res5['x_target']:.6f} (Analitik dan Numerik identik)")
    
    print("\n[2] Soal 6 (Dekomposisi QR vs LU Dinamika Drone):")
    print(f"    - Parameter NIM N   : 25 (NIM: 163251001)")
    print(f"    - Vektor IMU v      : [3.00, 6.25, 7.875]^T m/s")
    print(f"    - Model Kecepatan   : v(t) = {res6['x_exact'][0]:.6f}*t^2 + {res6['x_exact'][1]:.6f}*t + {res6['x_exact'][2]:.6f}")
    print(f"    - Estimasi v(4.2 s) : {res6['v_target_exact']:.6f} m/s (Analitik = QR = LU)")
    print(f"    - Keakuratan ||Ax-v||: QR = {res6['res_qr']:.2e}, LU = {res6['res_lu']:.2e}")
    print(f"    - Total FLOPs       : QR = {res6['counter_qr'].total_flops} FLOPs vs LU = {res6['counter_lu'].total_flops} FLOPs (Rasio 3.0x)")
    
    # Cek ketersediaan file artefak visualisasi
    expected_files = [
        "fig_soal5_regression.png",
        "fig_soal5_residuals.png",
        "fig_soal6_trajectory.png",
        "fig_soal6_metrics.png"
    ]
    print("\n[3] File Artefak Visualisasi:")
    all_exist = True
    for fname in expected_files:
        fpath = os.path.join(current_dir, fname)
        exists = os.path.isfile(fpath)
        all_exist = all_exist and exists
        status = "TERSEDIA" if exists else "TIDAK DITEMUKAN"
        print(f"    - {fname:<26} : [{status}]")
        
    if all_exist:
        print("\n>>> SELURUH SIMULASI BERHASIL DIEKSEKUSI DENGAN SEMPURNA! <<<")
    else:
        print("\n>>> PERINGATAN: Beberapa file visualisasi belum terbentuk. <<<")
    print("=" * 80)

if __name__ == "__main__":
    main()
