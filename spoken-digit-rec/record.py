import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 176400
CHUNK = 512
RECORD_SECONDS = 15
WAVE_OUTPUT_FILENAME = "./data/tests/result.wav"
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(
    format=FORMAT, 
    channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK
)
print("recording for 15 seconds")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# save results
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
