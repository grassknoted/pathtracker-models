import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of hGRU")

# parser.add_argument('train_list', type=str)
# parser.add_argument('val_list', type=str)

parser.add_argument('--name', type=str, default="hgru")
parser.add_argument('--model', type=str, default="hgru")
parser.add_argument('--algo', type=str, default="bptt")
parser.add_argument('--penalty', default=False, action='store_true')
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dist', type=int)
parser.add_argument('--thickness', type=int, help='thickness of the static occluder')
parser.add_argument('--length', type=int)
parser.add_argument('--gifs', type=int, help='Number of gifs to produce.')
parser.add_argument('--not_strict', default=True, action='store_false', help='strict ckpt restore or not')
parser.add_argument('--set_name', type=str, default=None)
parser.add_argument('--which_tests', type=str, default="all")
parser.add_argument('--save_rnn', default=False, action='store_true')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')


parser.add_argument('-d', '--dimensions', default=32, type=int)
parser.add_argument('-k', '--fb_kernel_size', default=7, type=int)


# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('-parallel', '--parallel', default= False, action='store_true',
                    help='Wanna parallelize the training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--log', default=False, action='store_true')

parser.add_argument('--val-freq', '-vf', default=2000, type=int,
                    metavar='N', help='Validation frequency')

