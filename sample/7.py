# 7. 減算合成1（ハイハット）
# 原音をフィルタで削ることで音色をつくる音響合成のテクニックを減算合成と呼ぶ。
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# 白色雑音からハイハットの音をつくることができることを確認しなさい．


import numpy as np
from scipy.io import wavfile
from IPython.display import display, Audio


def HPF(fs, fc, Q):
    fc /= fs
    fc = np.tan(np.pi * fc) / (2.0 * np.pi)
    a = np.zeros(3)
    b = np.zeros(3)
    a[0] = 1.0 + 2.0 * np.pi * fc / Q + 4.0 * np.pi * np.pi * fc * fc
    a[1] = (8.0 * np.pi * np.pi * fc * fc - 2.0) / a[0]
    a[2] = (1.0 - 2.0 * np.pi * fc / Q + 4.0 * np.pi * np.pi * fc * fc) / a[0]
    b[0] = 1.0 / a[0]
    b[1] = -2.0 / a[0]
    b[2] = 1.0 / a[0]
    a[0] = 1.0
    return a, b


def ADSR(fs, A, D, S, R, gate, duration):
    A = int(fs * A)
    D = int(fs * D)
    R = int(fs * R)
    gate = int(fs * gate)
    duration = int(fs * duration)
    e = np.zeros(duration)
    if A != 0:
        for n in range(A):
            e[n] = 1.0 - np.exp(-5.0 * n / A)

    if D != 0:
        for n in range(A, gate):
            e[n] = S + (1.0 - S) * np.exp(-5.0 * (n - A) / D)

    else:
        for n in range(A, gate):
            e[n] = S

    if R != 0:
        for n in range(gate, duration):
            e[n] = e[gate - 1] * np.exp(-5.0 * (n - gate + 1) / R)

    return e


def hihat_cymbal_close(fs, velocity, gate):
    duration = 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = (np.random.rand() * 2.0) - 1.0

    VCF_A = [0]
    VCF_D = [0]
    VCF_S = [1]
    VCF_R = [0]
    VCF_offset = [10000]
    VCF_depth = [0]

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], gate, duration)
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = HPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = [0]
    VCA_D = [0.1]
    VCA_S = [0]
    VCA_R = [0.1]
    VCA_offset = [0]
    VCA_depth = [1]

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], gate, duration)
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    gain = velocity / 127 / np.max(np.abs(s1))
    s1 *= gain

    return s1


def percussion(fs, note_number, velocity, gate):
    if note_number == 36:
        s = bass_drum(fs, velocity, gate)
    elif note_number == 40:
        s = snare_drum(fs, velocity, gate)
    elif note_number == 42:
        s = hihat_cymbal_close(fs, velocity, gate)

    return s


fs = 44100

note_number = 42
velocity = 100
gate = 0.1

s = percussion(fs, note_number, velocity, gate)
length_of_s = len(s)

for n in range(length_of_s):
    s[n] = (s[n] + 1.0) / 2.0 * 65536.0
    if s[n] > 65535.0:
        s[n] = 65535.0
    elif s[n] < 0.0:
        s[n] = 0.0
    s[n] = (s[n] + 0.5) - 32768

wavfile.write('7_hihat.wav', fs, s.astype(np.int16))

Audio('7_hihat.wav')
