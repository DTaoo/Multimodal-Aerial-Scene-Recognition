from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import time
import librosa
from PIL import Image, ImageEnhance
import os
import random

normalizer = np.load('../data/audio_feature_normalizer.npy')
mu    = normalizer[0]
sigma = normalizer[1]

def audio_extract(wav_file, sr =16000):
    wav = librosa.load(wav_file, sr=sr)[0]
    # Takes a waveform (length 160,000, sampling rate 16,000) and extracts filterbank features (size 400 * 64)
    spec = librosa.core.stft(wav, n_fft = 4096,
                             hop_length = 400, win_length = 1024,
                             window = 'hann', center = True, pad_mode = 'constant')
    mel = librosa.feature.melspectrogram(S = np.abs(spec), sr = sr, n_mels = 64, fmax = 8000)
    logmel = librosa.core.power_to_db(mel[:, :400])

    return logmel.T.astype('float32')


def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image


class CVSDataset(Dataset):
    def __init__(self, data_dir, data_sample, data_label, seed, enhance=True, use_KD=True, event_label_name='event_label_bayes_59'): # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.data_label = data_label
        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.use_KD = use_KD
        self.event_label_name = event_label_name
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        # return (img_data,audio_data,label)
        #np.random.seed(self.seed)
        #np.random.shuffle(self.index_list)

        image_path = os.path.join(self.data_dir, 'Dataset_v3_vision', self.data_sample[self.index_list[item]]+'.jpg')
        sound_path = os.path.join(self.data_dir, 'Dataset_v3_sound', self.data_sample[self.index_list[item]]+'.wav')

        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if self.enable_enhancement:
            image = augment_image(image)

        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2,0,1))
        #print(image.shape)
        sound = audio_extract(sound_path)
        sound = ((sound - mu) / sigma).astype('float32')

        # TODO: sound normalization

        sound = np.expand_dims(sound, 0)
        #sound = ((sound - mu) / sigma).astype('float32')

        scene_label = self.data_label[self.index_list[item]]

        if self.use_KD:
            event_path = os.path.join(self.data_dir, self.event_label_name, self.data_sample[self.index_list[item]]+'_ser.npy')
            event_label = np.load(event_path)

            corr_path = os.path.join('prior_knowledge_pca', 'salient_event_for_%d.npy' % scene_label)
            silent_corr = np.load(corr_path)
            silent_corr = silent_corr[0,:]
            return np.asarray(image).astype(float),np.asarray(sound).astype(float), scene_label, event_label, silent_corr
        else:
            return np.asarray(image).astype(float),np.asarray(sound).astype(float), scene_label

class CVS_Audio(Dataset):
    def __init__(self, data_dir, data_sample): # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample       

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, item):

        sound_path = os.path.join(self.data_dir, 'Dataset_v3_sound', self.data_sample[item]+'.wav')

        sound = audio_extract(sound_path)
        sound = ((sound - mu) / sigma).astype('float32')

        sound = np.expand_dims(sound, 0)

        return np.asarray(sound).astype(float)#, self.data_sample[item]




