# 加算合成2（パイプオルガン）
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# パイプオルガンの音色でカノンの演奏が聞こえてくることを確かめてください．
# 残響音をreverb関数を使って定義していることに注意してください．

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio


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


def pipe_organ(fs, note_number, velocity, gate):
    f0 = 440 * np.power(2, (note_number - 69) / 12)

    number_of_partial = 16

    VCO_A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    VCO_D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    VCO_S = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    VCO_R = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    VCO_offset = [f0 * 1, f0 * 2, f0 * 3, f0 * 4, f0 * 5, f0 * 6, f0 * 7, f0 *
                  8, f0 * 9, f0 * 10, f0 * 11, f0 * 12, f0 * 13, f0 * 14, f0 * 15, f0 * 16]
    VCO_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    VCA_A = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    VCA_D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    VCA_S = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    VCA_R = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    VCA_offset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    VCA_depth = [1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8,
                 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]

    duration = gate + 0.1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], gate, duration)
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        if np.max(vco) < fs / 2:
            x = np.zeros(length_of_s)
            t = 0
            for n in range(length_of_s):
                x[n] = np.sin(2 * np.pi * t)
                delta = vco[n] / fs
                t += delta
                if t >= 1:
                    t -= 1

            vca = ADSR(fs, VCA_A[i], VCA_D[i], VCA_S[i],
                       VCA_R[i], gate, duration)
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            for n in range(length_of_s):
                s[n] += x[n] * vca[n]

    gain = velocity / 127 / np.max(np.abs(s))
    s *= gain

    return s


def reverb(fs, x):
    length_of_x = len(x)

    d1 = int(fs * 0.03985)
    g1 = 0.871402
    u1 = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d1 >= 0:
            u1[n] = x[n - d1] + g1 * u1[n - d1]

    d2 = int(fs * 0.03610)
    g2 = 0.882762
    u2 = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d2 >= 0:
            u2[n] = x[n - d2] + g2 * u2[n - d2]

    d3 = int(fs * 0.03327)
    g3 = 0.891443
    u3 = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d3 >= 0:
            u3[n] = x[n - d3] + g3 * u3[n - d3]

    d4 = int(fs * 0.03015)
    g4 = 0.901117
    u4 = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d4 >= 0:
            u4[n] = x[n - d4] + g4 * u4[n - d4]

    v1 = np.zeros(length_of_x)
    for n in range(length_of_x):
        v1[n] = u1[n] + u2[n] + u3[n] + u4[n]

    d5 = int(fs * 0.005)
    g5 = 0.7
    u5 = np.zeros(length_of_x)
    v2 = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d5 >= 0:
            u5[n] = v1[n - d5] + g5 * u5[n - d5]

        v2[n] = u5[n] - g5 * (v1[n] + g5 * u5[n])

    d6 = int(fs * 0.0017)
    g6 = 0.7
    u6 = np.zeros(length_of_x)
    y = np.zeros(length_of_x)
    for n in range(length_of_x):
        if n - d6 >= 0:
            u6[n] = v2[n - d6] + g6 * u6[n - d6]

        y[n] = u6[n] - g6 * (v2[n] + g6 * u6[n])

    return y


score = np.array([[1, 1920, 76, 100, 960],
                  [1, 2880, 74, 100, 960],
                  [1, 3840, 72, 100, 960],
                  [1, 4800, 71, 100, 960],
                  [1, 5760, 69, 100, 960],
                  [1, 6720, 67, 100, 960],
                  [1, 7680, 69, 100, 960],
                  [1, 8640, 71, 100, 960],
                  [2, 1920, 60, 100, 960],
                  [2, 2880, 55, 100, 960],
                  [2, 3840, 57, 100, 960],
                  [2, 4800, 52, 100, 960],
                  [2, 5760, 53, 100, 960],
                  [2, 6720, 48, 100, 960],
                  [2, 7680, 53, 100, 960],
                  [2, 8640, 55, 100, 960]])

division = 480
tempo = 120
number_of_track = 2
end_of_track = 10
number_of_note = score.shape[0]

fs = 44100
length_of_s = int(fs * (end_of_track + 2))
track = np.zeros((length_of_s, number_of_track))
s = np.zeros(length_of_s)

for i in range(number_of_note):
    j = int(score[i, 0] - 1)
    onset = (score[i, 1] / division) * (60 / tempo)
    note_number = score[i, 2]
    velocity = score[i, 3]
    gate = (score[i, 4] / division) * (60 / tempo)
    x = pipe_organ(fs, note_number, velocity, gate)
    offset = int(fs * onset)
    length_of_x = len(x)
    for n in range(length_of_x):
        track[offset + n, j] += x[n]

for j in range(number_of_track):
    for n in range(length_of_s):
        s[n] += track[n, j]

s = reverb(fs, s)

master_volume = 0.5
s /= np.max(np.abs(s))
s *= master_volume

for n in range(length_of_s):
    s[n] = (s[n] + 1.0) / 2.0 * 65536.0
    if s[n] > 65535.0:
        s[n] = 65535.0
    elif s[n] < 0.0:
        s[n] = 0.0
    s[n] = (s[n] + 0.5) - 32768

wavfile.write('5.wav', fs, s.astype(np.int16))

Audio('5.wav')
