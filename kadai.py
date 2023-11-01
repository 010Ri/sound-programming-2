import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import spectrogram
from IPython.display import display, Audio
from pprint import pprint


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


def apply_envelope(note, attack_time, decay_time, release_time, fs):
    length = len(note)
    attack_samples = int(attack_time * fs)
    decay_samples = int(decay_time * fs)
    release_samples = int(release_time * fs)

    envelope = np.ones(length)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    envelope[attack_samples:(attack_samples + decay_samples)
             ] = np.linspace(1, 0.7, decay_samples)

    return note * envelope


def play_music(score):
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
        # エンベロープを適用
        attack_time = 0.1
        decay_time = 0.05
        release_time = 0.05
        xe = apply_envelope(x, attack_time, decay_time, release_time, fs)
        offset = int(fs * onset)
        length_of_x = len(x)
        for n in range(length_of_x):
            track[offset + n, j] += x[n]

    # オリジナルとエンベロープ付きの波形を可視化して確認
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title("オリジナル波形")
    plt.subplot(2, 1, 2)
    plt.plot(xe)
    plt.title("エンベロープ付き波形")
    plt.show()

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

    wavfile.write('kadai.wav', fs, s.astype(np.int16))
    return Audio('kadai.wav')


def analyze_wav_file(filename):
    sounds = AudioSegment.from_file(filename, 'wav')
    sample_rate, samples = wavfile.read(filename)

    # print(f'channel: {sounds.channels}')
    # print(f'frame rate: {sounds.frame_rate}')
    # print(f'duration: {sounds.duration_seconds} s')

    frequencies, times, spec = spectrogram(samples, fs=sample_rate)

    sig = np.array(sounds.get_array_of_samples())[::sounds.channels]
    dt = 1.0 / sounds.frame_rate

    tms = 0.0
    tme = sounds.duration_seconds
    tm = np.linspace(tms, tme, len(sig), endpoint=False)

    N = len(sig)
    X = np.fft.fft(sig)
    f = np.fft.fftfreq(N, dt)

    fig, (ax01, ax02, ax03) = plt.subplots(nrows=3, figsize=(7, 9))

    ax01.set_xlim(tms, tme)
    ax01.set_xlabel('time (s)')
    ax01.set_ylabel('x')
    ax01.plot(tm, sig)

    ax02.set_xlim(0, 2000)
    ax02.set_xlabel('frequency (Hz)')
    ax02.set_ylabel('|X|/N')
    ax02.plot(f[0:N // 2], np.abs(X[0:N // 2]) / N)

    amplitude_spectrum = np.abs(X[0:N // 2]) / N

    max_amplitude_frequencies = []

    for _ in range(7):
        max_amplitude_index = np.argmax(amplitude_spectrum)
        max_amplitude_frequency = f[max_amplitude_index]
        max_amplitude_value = np.abs(X[max_amplitude_index] / N)

        skip_range = int(10 / (f[1] - f[0]))
        amplitude_spectrum[max_amplitude_index -
                           skip_range:max_amplitude_index + skip_range + 1] = 0
        max_amplitude_frequencies.append(
            (max_amplitude_frequency, max_amplitude_value))

    # pprint(max_amplitude_frequencies)

    eps = 1e-10
    Z = 10.0 * np.log10(spec + eps)
    image = ax03.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    ax03.set_xlabel('Time (sec)')
    ax03.set_ylabel('Frequency (Hz)')
    colorbar = plt.colorbar(image, ax=ax03)

    # plt.show()

    return max_amplitude_frequencies


# WAVファイルの解析を実行
wav_filenames = [
    './piano/maou_se_inst_piano2_1do.wav',
    './piano/maou_se_inst_piano2_2re.wav',
    './piano/maou_se_inst_piano2_3mi.wav',
    './piano/maou_se_inst_piano2_4fa.wav',
    './piano/maou_se_inst_piano2_5so.wav',
    './piano/maou_se_inst_piano2_6ra.wav',
    './piano/maou_se_inst_piano2_7si.wav'
]

amplitude_arrays = []

for filename in wav_filenames:
    max_amplitude_frequencies = analyze_wav_file(filename)
    amplitude_arrays.append(max_amplitude_frequencies)

# print(amplitude_arrays)
# pprint(amplitude_arrays)

normalized_amplitude_arrays = []

for amplitudes in amplitude_arrays:
    normalized_amplitudes = [
        (f, v / amplitudes[0][1]) for f, v in amplitudes]
    normalized_amplitude_arrays.append(normalized_amplitudes)

print("\nmaxを1に")
pprint(normalized_amplitude_arrays)

# 自動演奏を実行
score = []

i = 2  # ミ5
for j in range(7):
    score.append([1, 2, normalized_amplitude_arrays[i][j][0] * 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 1  # レ5
for j in range(7):
    score.append([1, 3, normalized_amplitude_arrays[i][j][0] * 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 0  # ド5
for j in range(7):
    score.append([1, 4, normalized_amplitude_arrays[i][j][0] * 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 6  # シ4
for j in range(7):
    score.append([1, 5, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

i = 5  # ラ4
for j in range(7):
    score.append([1, 6, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

i = 4  # ソ4
for j in range(7):
    score.append([1, 7, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

i = 5  # ラ4
for j in range(7):
    score.append([1, 8, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

i = 6  # シ4
for j in range(7):
    score.append([1, 9, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

# ここからトラック2
i = 0  # ド4
for j in range(7):
    score.append([2, 2, normalized_amplitude_arrays[i][j][0],
                 normalized_amplitude_arrays[i][j][1], 1])

i = 4  # ソ3
for j in range(7):
    score.append([2, 3, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 5  # ラ3
for j in range(7):
    score.append([2, 4, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 2  # ミ3
for j in range(7):
    score.append([2, 5, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 3  # ファ3
for j in range(7):
    score.append([2, 6, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 0  # ド4
for j in range(7):
    score.append([2, 7, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 3  # ファ3
for j in range(7):
    score.append([2, 8, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])

i = 4  # ソ3
for j in range(7):
    score.append([2, 9, normalized_amplitude_arrays[i][j][0] / 2,
                 normalized_amplitude_arrays[i][j][1], 1])


score = np.array(score)  # PythonリストをNumPy配列に変換

# print(score)
amplitude_spectrum = play_music(score)


# score = np.array([[1, 2, 659.26, 0.8, 1],  # ミ5
#                   [1, 3, 587.33, 0.8, 1],  # レ5
#                   [1, 4, 523.25, 0.8, 1],  # ド5
#                   [1, 5, 493.88, 0.8, 1],  # シ4
#                   [1, 6, 440.00, 0.8, 1],  # ラ4
#                   [1, 7, 392.00, 0.8, 1],  # ソ4
#                   [1, 8, 440.00, 0.8, 1],  # ラ4
#                   [1, 9, 493.88, 0.8, 1],  # シ4
#                   [2, 2, 261.63, 0.8, 1],  # ド4
#                   [2, 3, 196.00, 0.8, 1],  # ソ3
#                   [2, 4, 220.00, 0.8, 1],  # ラ3
#                   [2, 5, 164.81, 0.8, 1],  # ミ3
#                   [2, 6, 174.61, 0.8, 1],  # ファ3
#                   [2, 7, 130.81, 0.8, 1],  # ド3
#                   [2, 8, 174.61, 0.8, 1],  # ファ3
#                   [2, 9, 196.00, 0.8, 1]])  # ソ3
