from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob

DATA_DIRECTORY = './data/spoken_numbers_pcm/'
empty = AudioSegment.silent(1000)
fnames = glob.glob(DATA_DIRECTORY + "*.wav")

for fname in fnames:
    audioFile = AudioSegment.from_wav(fname)
    audioFile = empty + audioFile + empty

    chunk = split_on_silence(audioFile, min_silence_len=150, silence_thresh=-50)[0]
    chunk.export(fname, format='wav')
