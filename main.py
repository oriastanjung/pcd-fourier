import numpy as np

def TransformasiFourier2Dimensi(f):
    # Mendapatkan ukuran matriks spasial
    M, N = f.shape
    
    # Inisialisasi matriks frekuensi kompleks
    F = np.zeros((M, N), dtype=np.complex128)

    # Iterasi untuk setiap koordinat frekuensi (u, v)
    for u in range(M):
        for v in range(N):
            # Iterasi untuk setiap elemen matriks spasial (x, y)
            for x in range(M):
                for y in range(N):
                    # Hitung kontribusi masing-masing elemen pada F(u, v)
                    F[u, v] += f[x, y] * np.exp(-1j * 2 * np.pi * ((u * x) / M + (v * y) / N))

    return F

# Matriks spasial
f = np.array([
    [1, 3, 5, 2, 7],
    [2, 4, 5, 0, 0],
    [2, 3, 1, 6, 6],
    [2, 3, 1, 0, 7],
    [5, 4, 4, 1, 0]
])


# Hitung Transformasi Fourier Diskrit 2D (DFT)
result = TransformasiFourier2Dimensi(f)

# Tampilkan hasil matrix frekuensi
print("Matriks spasial f(x, y):")
print(f)
print("\nTransformasi Fourier Diskrit 2D (DFT):")
print(result)

def inverse_DFT2D(F):
    M, N = F.shape
    f = np.zeros((M, N), dtype=np.complex128)

    for x in range(M):
        for y in range(N):
            for u in range(M):
                for v in range(N):
                    f[x, y] += F[u, v] * np.exp(1j * 2 * np.pi * ((u * x) / M + (v * y) / N))

    return f / (M * N)

# Menghitung invers DFT dari hasil sebelumnya
reconstructed_f = inverse_DFT2D(result)

# Tampilkan hasil
print("Matriks spasial yang diperoleh kembali:")
print(np.real(reconstructed_f))
