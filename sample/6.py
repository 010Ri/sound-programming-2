# 6. フィルタ
# フィルタは、特定の帯域の周波数成分だけを選択的に通過させるふるいである．
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# 高域の周波数成分がカットされることを確認しなさい．

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


fs = 8000

length_of_s = int(fs * 1)
s0 = np.zeros(length_of_s)

for i in range(1, 9):
    for n in range(length_of_s):
        s0[n] += 1.0 * np.sin(2 * np.pi * 440 * i * n / fs)

gain = 0.5 / np.max(np.abs(s0))
s0 *= gain

for n in range(length_of_s):
    s0[n] = (s0[n] + 1.0) / 2.0 * 65536.0
    if s0[n] > 65535.0:
        s0[n] = 65535.0
    elif s0[n] < 0.0:
        s0[n] = 0.0
    s0[n] = (s0[n] + 0.5) - 32768

wavfile.write('6_pulse_train.wav', fs, s0.astype(np.int16))

Audio('6_pulse_train.wav')

s1 = np.zeros(length_of_s)

fc = 880
Q = 1 / np.sqrt(2)
a, b = LPF(fs, fc, Q)
for n in range(length_of_s):
    for m in range(0, 3):
        if n - m >= 0:
            s1[n] += b[m] * s0[n - m]

    for m in range(1, 3):
        if n - m >= 0:
            s1[n] += -a[m] * s1[n - m]

master_volume = 0.5
s1 /= np.max(np.abs(s1))
s1 *= master_volume

for n in range(length_of_s):
    s1[n] = (s1[n] + 1.0) / 2.0 * 65536.0
    if s1[n] > 65535.0:
        s1[n] = 65535.0
    elif s1[n] < 0.0:
        s1[n] = 0.0
    s1[n] = (s1[n] + 0.5) - 32768

wavfile.write('6.wav', fs, s1.astype(np.int16))

Audio('6.wav')
