import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = "XXXXXXXXXXXsample/happy-piano/ours/scale-0.1.wav"

y, sr = librosa.load(audio_path)
# 提取mel频谱
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

# 将频谱转换为分贝单位
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 绘制频谱图并改变颜色
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.tight_layout()
plt.show()