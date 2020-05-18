import sys
sys.path.append("../data/")
from data_partition import data_construction
import argparse
import os
import librosa
import numpy
import pickle

GAS_FEATURE_DIR = '/mnt/scratch/hudi/sound-dataset/audioset'

with open(os.path.join(GAS_FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    mu, sigma = u.load()

def extract(wav):
    # Takes a waveform (length 160,000, sampling rate 16,000) and extracts filterbank features (size 400 * 64)
    spec = librosa.core.stft(wav, n_fft = 4096,
                             hop_length = 400, win_length = 1024,
                             window = 'hann', center = True, pad_mode = 'constant')
    mel = librosa.feature.melspectrogram(S = numpy.abs(spec), sr=16000, n_mels = 64, fmax = 8000)
    logmel = librosa.core.power_to_db(mel[:, :400])
    return logmel.T.astype('float32')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sound_feature_extraction')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/soundscape/data/',
                        help='image net weights')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    args = parser.parse_args()

    (train_sample, train_label, test_sample, test_label) = data_construction(args.data_dir)

    train_sample_num = len(train_sample)
    audio_fea = []
    for i in range(train_sample_num):
        if i % 100 == 0:
            print('prcessing %d...' % i)
        sound_path = os.path.join(args.data_dir, 'Dataset_v3_sound', train_sample[i] + '.wav')
        wav = librosa.load(sound_path, sr=16000)[0]
        fea = extract(wav)
        if audio_fea == []:
            audio_fea = fea
        else:
            audio_fea = numpy.concatenate((audio_fea, fea), axis=0)

    mu = numpy.mean(audio_fea, axis=0)
    sigma = numpy.std(audio_fea, axis=0)
    normalizer = []
    normalizer.append(mu)
    normalizer.append(sigma)
    numpy.save('audio_feature_normalizer', normalizer)




