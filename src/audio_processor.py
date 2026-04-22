import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Dependencies: pip install librosa matplotlib numpy
# Dependencies: ffmpeg must be installed and in PATH for librosa to load audio from video files
## OR convert to .wav audio using vlc or ffmpeg CLI before running this script

in_file = input("Enter the path to the audio file: ")
y, sr = librosa.load(in_file, sr=None)

# Filter out weak signals, only keep samples above this amplitude
# Raise this if still getting false positives in quiet sections
MIN_AMPLITUDE = 0.15

# Detect onsets with backtrack=True so lines snap to the actual attack
onset_frames = librosa.onset.onset_detect(
    y=y, sr=sr,
    backtrack=True,       # snap to the real start of each note
    delta=0.3,            # higher = stricter, fewer detections
    wait=10               # minimum frames between onsets (avoids clusters)
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Remove onsets where the audio around that time is basically silent
def amplitude_at(t, y, sr, window=0.05):
    start = int(t * sr)
    end   = int((t + window) * sr)
    return np.max(np.abs(y[start:end])) if end < len(y) else 0

onset_times = [t for t in onset_times if amplitude_at(t, y, sr) > MIN_AMPLITUDE]

# C major scale up and down -- hardcoded for this demo
CMAJOR = ["C", "D", "E", "F", "G", "A", "B", "C",
           "B", "A", "G", "F", "E", "D", "C"]

fig, ax = plt.subplots(figsize=(13, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax)

for i, t in enumerate(onset_times):
    ax.axvline(t, color='red', linewidth=1.5)
    if i < len(CMAJOR):
        ax.text(t + 0.05, 0.85, CMAJOR[i],
                transform=ax.get_xaxis_transform(),
                fontsize=9, color='red', fontweight='bold')

ax.set_title("Audio waveform with detected note onsets — C Major Scale")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude (normalized -1 to 1)")
ax.legend(["Note onsets"], loc="upper right")
plt.tight_layout()
plt.savefig("onsets.png", dpi=150)
plt.show()

print(f"Detected {len(onset_times)} onsets")
for i, t in enumerate(onset_times):
    note = CMAJOR[i] if i < len(CMAJOR) else "?"
    print(f"  {note}: {t:.2f}s")