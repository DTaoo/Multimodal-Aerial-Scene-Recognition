import os
import librosa
import numpy as np
import numpy as np
import shutil

CSV_Dataset = 'F:/download/CVS_Dataset_New/'
ffmpeg_path = 'F:/ffmpeg/bin/ffmpeg'
audio_files = os.listdir(CSV_Dataset+'audio')

def extract(wav_file):
    wav = librosa.load(wav_file)[0]
    # Takes a waveform (length 160,000, sampling rate 16,000) and extracts filterbank features (size 400 * 64)
    spec = librosa.core.stft(wav, n_fft = 4096,
                             hop_length = 400, win_length = 1024,
                             window = 'hann', center = True, pad_mode = 'constant')
    mel = librosa.feature.melspectrogram(S = np.abs(spec), sr = 16000, n_mels = 64, fmax = 8000)
    logmel = librosa.core.power_to_db(mel[:, :400])
    return logmel.T.astype('float32')

for i,a in enumerate(audio_files):

    if a.endswith('mp3'):
        print('%d/%d' % (i, len(audio_files)))
        id = a.replace('.mp3','')
        save_path = CSV_Dataset + 'audio_feat/' + id + '.npy'
        if id == '12122':
            print('x')
        if '_' in id:
            clear_id = id.split('_')[0]
            clear_path = CSV_Dataset + 'audio_feat/' + clear_id + '.npy'
            if os.path.exists(clear_path) and (not os.path.exists(save_path)):
                shutil.copy(clear_path, save_path)
                continue



        if os.path.exists(save_path):
            continue
        cmd = ffmpeg_path+' -i ' + CSV_Dataset+'audio/'+a + ' '+CSV_Dataset+ 'audio_wav/' + id+'.wav'
        os.system(cmd)


        feat = extract(CSV_Dataset+ 'audio_wav/' + id+'.wav')

        channel_feat = np.asarray([feat])

        np.save(save_path,channel_feat)

        # remove .wav
        os.remove(CSV_Dataset+ 'audio_wav/' + id +'.wav')





