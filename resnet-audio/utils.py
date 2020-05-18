import librosa
import numpy
import sys, os, os.path, glob
import _pickle as cPickle
import pickle
from scipy.io import loadmat
from scipy import stats
import numpy
from multiprocessing import Process, Queue
import torch
from torch.autograd import Variable

N_CLASSES = 527
N_WORKERS = 6

GAS_FEATURE_DIR = '/mnt/scratch/hudi/sound-dataset/audioset'

#with open(os.path.join(GAS_FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
#    mu, sigma = pickle.load(f)

with open(os.path.join(GAS_FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    mu, sigma = u.load()


def sample_generator(file_list, random_seed = 15213):
    rng = numpy.random.RandomState(random_seed)
    while True:
        rng.shuffle(file_list)
        for filename in file_list:
            data = loadmat(filename)
            feat = ((data['feat'] - mu) / sigma).astype('float32')
            feat = numpy.expand_dims(feat,1)
            labels = data['labels'].astype('float32')
            for i in range(len(data['feat'])):
                yield feat[i], labels[i]

def worker(queues, file_lists, random_seed):
    generators = [sample_generator(file_lists[i], random_seed + i) for i in range(len(file_lists))]
    while True:
        for gen, q in zip(generators, queues):
            q.put(next(gen))

def batch_generator(batch_size, random_seed = 15213):
    queues = [Queue(5) for class_id in range(N_CLASSES)]
    file_lists = [sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, 'GAS_train_unbalanced_class%03d_part*.mat' % class_id))) for class_id in range(N_CLASSES)]

    for worker_id in range(N_WORKERS):
        p = Process(target = worker, args = (queues[worker_id::N_WORKERS], file_lists[worker_id::N_WORKERS], random_seed))
        p.daemon = True
        p.start()

    rng = numpy.random.RandomState(random_seed)
    batch = []
    while True:
        rng.shuffle(queues)
        for q in queues:
            batch.append(q.get())
            if len(batch) == batch_size:
                yield tuple(Variable(torch.from_numpy(numpy.stack(x))).cuda() for x in zip(*batch))
                batch = []

def bulk_load(prefix):
    feat = []; labels = []; hashes = []
    for filename in sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, '%s_*.mat' % prefix))):
        data = loadmat(filename)
        fea = ((data['feat'] - mu) / sigma).astype('float32')
        feat.append(numpy.expand_dims(fea, 1))
        labels.append(data['labels'].astype('bool'))
        hashes.append(data['hashes'])
    return numpy.concatenate(feat), numpy.concatenate(labels), numpy.concatenate(hashes)


###### evaluation ########
def roc(pred, truth):
    data = numpy.array(sorted(zip(pred, truth), reverse = True))
    pred, truth = data[:,0], data[:,1]
    TP = truth.cumsum()
    FP = (1 - truth).cumsum()
    mask = numpy.concatenate([numpy.diff(pred) < 0, numpy.array([True])])
    TP = numpy.concatenate([numpy.array([0]), TP[mask]])
    FP = numpy.concatenate([numpy.array([0]), FP[mask]])
    return TP, FP

def ap_and_auc(pred, truth):
    TP, FP = roc(pred, truth)
    auc = ((TP[1:] + TP[:-1]) * numpy.diff(FP)).sum() / (2 * TP[-1] * FP[-1])
    precision = TP[1:] / (TP + FP)[1:]
    weight = numpy.diff(TP)
    ap = (precision * weight).sum() / TP[-1]
    return ap, auc

def dprime(auc):
    return stats.norm().ppf(auc) * numpy.sqrt(2.0)

def gas_eval(pred, truth):
    if truth.ndim == 1:
        ap, auc = ap_and_auc(pred, truth)
    else:
        ap, auc = numpy.array([ap_and_auc(pred[:,i], truth[:,i]) for i in range(truth.shape[1]) if truth[:,i].any()]).mean(axis = 0)
    return ap, auc, dprime(auc)


def extract(wav):
    # Takes a waveform (length 160,000, sampling rate 16,000) and extracts filterbank features (size 400 * 64)
    spec = librosa.core.stft(wav, n_fft = 4096,
                             hop_length = 400, win_length = 1024,
                             window = 'hann', center = True, pad_mode = 'constant')
    mel = librosa.feature.melspectrogram(S = numpy.abs(spec), sr = 16000, n_mels = 64, fmax = 8000)
    logmel = librosa.core.power_to_db(mel[:, :400])
    return logmel.T.astype('float32')
