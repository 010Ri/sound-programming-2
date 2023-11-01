# 9. 減算合成3（スネアドラム）
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# 矩形波と白色雑音からスネアドラムの音をつくることができることを確認しなさい．

import numpy as np
from scipy.io import wavfile
from IPython.display import display, Audio


def LPF(fs, fc, Q):
    fc /= fs
    fc = np.tan(np.pi * fc) / (2.0 * np.pi)
    a = np.zeros(3)
    b = np.zeros(3)
    a[0] = 1.0 + 2.0 * np.pi * fc / Q + 4.0 * np.pi * np.pi * fc * fc
    a[1] = (8.0 * np.pi * np.pi * fc * fc - 2.0) / a[0]
    a[2] = (1.0 - 2.0 * np.pi * fc / Q + 4.0 * np.pi * np.pi * fc * fc) / a[0]
    b[0] = 4.0 * np.pi * np.pi * fc * fc / a[0]
    b[1] = 8.0 * np.pi * np.pi * fc * fc / a[0]
    b[2] = 4.0 * np.pi * np.pi * fc * fc / a[0]
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


def compressor(fs, x):
    length_of_s = len(x)

    gain = 1.0 / np.max(np.abs(x))
    x *= gain

    threshold = 0.6
    ratio = 1 / 8
    width = 0.2
    gain = 1 / (threshold + (1.0 - threshold) * ratio)

    for n in range(length_of_s):
        if x[n] < 0:
            sign_of_s = -1
        else:
            sign_of_s = 1

        abs_of_s = np.abs(x[n])

        if abs_of_s >= threshold - width / 2 and abs_of_s < threshold + width / 2:
            abs_of_s = abs_of_s + (ratio - 1) * (abs_of_s - threshold +
                                                 width / 2)*(abs_of_s - threshold + width / 2) / (width * 2)
        elif abs_of_s >= threshold + width / 2:
            abs_of_s = threshold + (abs_of_s - threshold) * ratio

        x[n] = sign_of_s * abs_of_s * gain

    return x


def snare_drum(fs, velocity, gate):
    duration = 1

    length_of_s = int(fs * duration)
    sa0 = np.zeros(length_of_s)
    sb0 = np.zeros(length_of_s)

    VCO_A = [0]
    VCO_D = [0]
    VCO_S = [1]
    VCO_R = [0]
    VCO_offset = [150]
    VCO_depth = [0]

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], duration, duration)
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        if x < 0.5:
            sa0[n] = 1
        else:
            sa0[n] = -1

        delta = vco[n] / fs

        if 1 - delta <= x and x < 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            sa0[n] += d
        elif 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            sa0[n] += d

        if 0.5 - delta <= x and x < 0.5:
            t = (x - 0.5) / delta
            d = t * t + 2 * t + 1
            sa0[n] -= d
        elif 0.5 <= x and x < 0.5 + delta:
            t = (x - 0.5) / delta
            d = -t * t + 2 * t - 1
            sa0[n] -= d

        x += delta
        if x >= 1:
            x -= 1

    np.random.seed(0)
    for n in range(length_of_s):
        sb0[n] = (np.random.rand() * 2.0) - 1.0

    VCA_A = [0]
    VCA_D = [0]
    VCA_S = [1]
    VCA_R = [0]
    VCA_offset = [0]
    VCA_depth = [1]

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], duration, duration)
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb0[n] *= vca[n]

    s0 = np.zeros(length_of_s)
    for n in range(length_of_s):
        s0[n] = sa0[n] * 0.3 + sb0[n] * 0.7

    VCF_A = [0]
    VCF_D = [0.1]
    VCF_S = [0]
    VCF_R = [0.1]
    VCF_offset = [8000]
    VCF_depth = [-7800]

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], gate, duration)
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = [0]
    VCA_D = [0.2]
    VCA_S = [0]
    VCA_R = [0.2]
    VCA_offset = [0]
    VCA_depth = [1]

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], gate, duration)
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    s1 = compressor(fs, s1)

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

note_number = 40
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

wavfile.write('9_snare_drum.wav', fs, s.astype(np.int16))

Audio('9_snare_drum.wav')
