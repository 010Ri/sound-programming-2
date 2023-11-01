# 2. 自動演奏
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# カノンの演奏が聞こえてくることを確かめてください．


from IPython.display import display, Audio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def sine_wave(fs, f, a, duration):
    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)
    for n in range(length_of_s):
        s[n] = np.sin(2 * np.pi * f * n / fs)

    for n in range(int(fs * 0.01)):
        s[n] *= n / (fs * 0.01)
        s[length_of_s - n - 1] *= n / (fs * 0.01)

    gain = a / np.max(np.abs(s))
    s *= gain
    return s


score = np.array([[1, 2, 659.26, 0.8, 1],
                  [1, 3, 587.33, 0.8, 1],
                  [1, 4, 523.25, 0.8, 1],
                  [1, 5, 493.88, 0.8, 1],
                  [1, 6, 440.00, 0.8, 1],
                  [1, 7, 392.00, 0.8, 1],
                  [1, 8, 440.00, 0.8, 1],
                  [1, 9, 493.88, 0.8, 1],
                  [2, 2, 261.63, 0.8, 1],
                  [2, 3, 196.00, 0.8, 1],
                  [2, 4, 220.00, 0.8, 1],
                  [2, 5, 164.81, 0.8, 1],
                  [2, 6, 174.61, 0.8, 1],
                  [2, 7, 130.81, 0.8, 1],
                  [2, 8, 174.61, 0.8, 1],
                  [2, 9, 196.00, 0.8, 1]])

number_of_track = 2
number_of_note = score.shape[0]

fs = 44100
length_of_s = int(fs * 12)
track = np.zeros((length_of_s, number_of_track))
s = np.zeros(length_of_s)

for i in range(number_of_note):
    j = int(score[i, 0] - 1)
    onset = score[i, 1]
    f = score[i, 2]
    a = score[i, 3]
    duration = score[i, 4]
    x = sine_wave(fs, f, a, duration)
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

wavfile.write('2.wav', fs, s.astype(np.int16))

Audio('2.wav')
