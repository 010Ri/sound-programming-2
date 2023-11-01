# 4. 加算合成1（鉄琴）
# 同じ曲でも，楽器を取り替えると，雰囲気は一変する．
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# 鉄琴の音色でカノンの演奏が聞こえてくることを確かめてください．
# 音の時間変化をADSR関数を使って定義していることに注意してください．


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


def glockenspiel(fs, note_number, velocity, gate):
    f0 = 440 * np.power(2, (note_number - 69) / 12)

    number_of_partial = 5

    VCO_A = [0, 0, 0, 0, 0]
    VCO_D = [0, 0, 0, 0, 0]
    VCO_S = [1, 1, 1, 1, 1]
    VCO_R = [0, 0, 0, 0, 0]
    VCO_offset = [f0 * 1, f0 * 2.76, f0 * 5.40, f0 * 8.93, f0 * 13.32]
    VCO_depth = [0, 0, 0, 0, 0]

    VCA_A = [0, 0, 0, 0, 0]
    VCA_D = [2, 0.5, 0.2, 0.2, 0.1]
    VCA_S = [0, 0, 0, 0, 0]
    VCA_R = [2, 0.5, 0.2, 0.2, 0.1]
    VCA_offset = [0, 0, 0, 0, 0]
    VCA_depth = [1, 0.5, 0.4, 0.4, 0.2]

    duration = 2

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
    x = glockenspiel(fs, note_number, velocity, gate)
    offset = int(fs * onset)
    length_of_x = len(x)
    for n in range(length_of_x):
        track[offset + n, j] += x[n]

for j in range(number_of_track):
    for n in range(length_of_s):
        s[n] += track[n, j]

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

wavfile.write('4.wav', fs, s.astype(np.int16))

Audio('4.wav')
