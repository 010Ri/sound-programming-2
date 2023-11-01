# 3. 楽譜データをMIDIのパラメータで書き換える
# MIDIは、音の高さをノートナンバー，音の大きさをベロシティ，音の長さをゲートタイムによって定義している．
# 「New」ボタンをクリックし，新しくウィンドウを作成しなさい．
# つづいて，以下のプログラムを順番にセルに貼りつけ，実行しなさい．
# カノンの演奏が聞こえてくることを確かめてください．


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio


def sine_wave(fs, note_number, velocity, gate):
    length_of_s = int(fs * gate)
    s = np.zeros(length_of_s)
    f = 440 * np.power(2, (note_number - 69) / 12)
    for n in range(length_of_s):
        s[n] = np.sin(2 * np.pi * f * n / fs)

    for n in range(int(fs * 0.01)):
        s[n] *= n / (fs * 0.01)
        s[length_of_s - n - 1] *= n / (fs * 0.01)

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
    x = sine_wave(fs, note_number, velocity, gate)
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

wavfile.write('3.wav', fs, s.astype(np.int16))

Audio('3.wav')
