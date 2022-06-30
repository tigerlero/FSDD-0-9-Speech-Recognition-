from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import random

data_dir = './data/16bit/'
empty = AudioSegment.silent(1000)
fnames = glob.glob(data_dir + "*.wav")
n = len(fnames) - 1
digits = random.randint(4, 10) # number of digits
output_dir = "./data/tests/output.wav"

# generate a wav file with random digits(4-10)
output = empty
for i in range(0, digits):
    idx = random.randint(0, n) # random index
    audio = AudioSegment.from_wav(fnames[idx]) # random audio
    output = output + audio + empty
output.export(output_dir, format='wav')
