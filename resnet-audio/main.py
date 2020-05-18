import sys, os, os.path, time
import argparse
import numpy
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from resnet_audio import resnet50
from utils import batch_generator, bulk_load, gas_eval

torch.backends.cudnn.benchmark = True

# Parse input arguments
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']

parser = argparse.ArgumentParser()
parser.add_argument('--kernel_size', type = str, default = '3')     # 'n' or 'nxm'
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--ckpt_size', type = int, default = 1000)      # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.8)
parser.add_argument('--max_ckpt', type = int, default = 30)
parser.add_argument('--random_seed', type = int, default = 15213)
args = parser.parse_args()

if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

numpy.random.seed(args.random_seed)

# Load data
train_gen = batch_generator(batch_size = args.batch_size, random_seed = args.random_seed)
gas_valid_x, gas_valid_y, _ = bulk_load('GAS_valid')
gas_eval_x, gas_eval_y, _ = bulk_load('GAS_eval')

# Build model
args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
model = resnet50().cuda()

if args.optimizer == 'sgd':
    optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr = args.init_lr)

scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.lr_factor, patience = args.lr_patience) if args.lr_factor < 1.0 else None
criterion = nn.BCELoss(reduce=True)


for checkpoint in range(1, args.max_ckpt + 1):
    # Train for args.ckpt_size batches
    model.train()
    train_loss = 0
    for batch in range(1, args.ckpt_size + 1):
        x, y = next(train_gen)
        optimizer.zero_grad()
        global_prob = model(x)
        global_prob.clamp_(min = 1e-7, max = 1 - 1e-7) # 0.~1.
        loss = criterion(global_prob, y)

        train_loss += loss.data
        #if numpy.isnan(train_loss) or numpy.isinf(train_loss): break
        loss.backward()
        optimizer.step()
        print('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % (checkpoint, batch, args.ckpt_size, train_loss / batch))
        #del x, y, global_prob, loss         # This line and next line: to save GPU memory
        #torch.cuda.empty_cache()            # I don't know if they're useful or not
    train_loss /= args.ckpt_size

    # Evaluate model
    model.eval()
    sys.stderr.write('Evaluating model on GAS_VALID ...\n')
    global_prob = model.predict(gas_valid_x)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_valid_y)
    print('Evaluating map is %.3f, auc is %.3f' % (gv_map, gv_mauc))

    # Abort if training has gone mad
    #if numpy.isnan(train_loss) or numpy.isinf(train_loss):
    #    break

    # Save model. Too bad I can't save the scheduler
    MODEL_PATH = 'model'
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % checkpoint)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    sys.stderr.write('Saving model to %s ...\n' % MODEL_FILE)
    torch.save(state, MODEL_FILE)

    # Update learning rate
    if scheduler is not None:
        scheduler.step(gv_map)
