#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

"""

import os
import re
import time
import torch
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
from torch import nn
import torch.optim
import numpy as np

# from utils.dataset import DataSetSeg
from utils import engine
from utils.TFRDataset import tfr_data_loader
from models.hgrucleanSEG import hConvGRU
from models.FFnet import FFConvNet
from models.ffhgru import FFhGRU  # , FFhGRUwithGabor, FFhGRUwithoutGaussian, FFhGRUdown
from models import ffhgru

from utils.transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint
from statistics import mean
from utils.opts import parser
from utils import presets
import matplotlib
# import imageio
from torch._six import inf
from torchvideotransforms import video_transforms, volume_transforms
from torchvision.models import video
from models import nostridetv as nostride_video
from tqdm import tqdm
from types import SimpleNamespace
from glob import glob


torch.backends.cudnn.benchmark = True
args = parser.parse_args()
video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False
disentangle_channels = False
plot_incremental = False
debug_data = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_best_model(directory, model, strict=True, prep_gifs=3, batch_size=100, save_rnn=False, which_tests="all"):
    """Given a directory, find the best performing checkpoint and evaluate it on all datasets."""
    # perfs = np.load(os.path.join(directory, "val.npz"))["loss"]
    # arg_perf = np.argmin(perfs)
    if os.path.exists(os.path.join(directory, "val.npz")):
        perfs = np.load(os.path.join(directory, "val.npz"))["balacc"]
        # perfs = np.load(os.path.join(directory, "val.npz"))["hp_dict"]
    elif os.path.exists(os.path.join(os.path.sep.join(directory.split(os.path.sep)[:-1]), "{}val.npz".format(directory.split(os.path.sep)[-1]))):
        perfs = np.load(os.path.join(os.path.sep.join(directory.split(os.path.sep)[:-1]), "{}val.npz".format(directory.split(os.path.sep)[-1])))["balacc"]
    elif os.path.exists(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results", "_{}".format(directory.split(os.path.sep)[1]), "{}val.npz".format(directory.split(os.path.sep)[-1]))):
        perfs = np.load(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results", "_{}".format(directory.split(os.path.sep)[1]), "{}val.npz".format(directory.split(os.path.sep)[-1])))["balacc"]
        directory = os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results", "_{}".format(directory.split(os.path.sep)[1]), "{}".format(directory.split(os.path.sep)[-1]))
    elif os.path.exists(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", "_{}".format(directory.split(os.path.sep)[1]), "{}val.npz".format(directory.split(os.path.sep)[-1]))):
        perfs = np.load(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", "_{}".format(directory.split(os.path.sep)[1]), "{}val.npz".format(directory.split(os.path.sep)[-1])))["balacc"]
        directory = os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", "_{}".format(directory.split(os.path.sep)[1]), "{}".format(directory.split(os.path.sep)[-1]))
    elif os.path.exists(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", directory.split(os.path.sep)[1], "{}val.npz".format(directory.split(os.path.sep)[-1]))):
        perfs = np.load(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", directory.split(os.path.sep)[1], "{}val.npz".format(directory.split(os.path.sep)[-1])))["balacc"]
        directory = os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new", directory.split(os.path.sep)[1], "{}".format(directory.split(os.path.sep)[-1]))
    else:
        print("Falling back to data cifs")
        perfs = np.load(os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results", directory.split(os.path.sep)[1], "{}val.npz".format(directory.split(os.path.sep)[-1])))["balacc"]
        directory = os.path.join("/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results", directory.split(os.path.sep)[1], "{}".format(directory.split(os.path.sep)[-1]))
    arg_perf = np.argmax(perfs)
    weights = glob(os.path.join(directory, "saved_models", "*.tar"))
    if not len(weights):
        weights = glob(os.path.join(directory, "*.tar"))
        perfs = np.asarray([int(re.search("acc_00\d\d", w).group().split("_")[1]) for w in weights])
        arg_perf = np.argsort(perfs)[-1]
        # hps = np.load(os.path.join(directory, "hp_dict.npz"))
    else:
        # hps = np.load(os.path.join(directory, "saved_models", "hp_dict.npz"))
        pass
    if not len(weights):
        import pdb;pdb.set_trace()
        weights.sort(key=os.path.getmtime)
    weights = np.asarray(weights)
    ckpt = weights[arg_perf]

    # Fix model name if needed
    model = engine.fix_model_name(model)
    print("Evaluating {}".format(model))

    # Construct new args
    args = SimpleNamespace()
    if "performer" in model.lower():
        # args.parallel = False
        args.batch_size = 8
    else:
        # args.parallel = True
        args.batch_size = batch_size
    if "hgru_" in model.lower():
        args.strict = False
    else:
        args.strict = strict
    check_weight = torch.load(ckpt)
    if 'state_dict' in check_weight:
        par_check = [x for x in torch.load(ckpt)['state_dict'].keys()][0]
    else:
        par_check = [x for x in torch.load(ckpt).keys()][0]
    if "module" in par_check:
        args.parallel = True
    else:
        args.parallel = False
    args.ckpt = ckpt
    args.model = model  # hps["model"]
    args.penalty = "Testing"
    args.algo = "Testing"
    args.save_rnn = save_rnn
    if "imagenet" in directory:
        args.pretrained = True
    else:
        args.pretrained = False
    if which_tests == "64":
        ds = engine.get_64_gen()
    elif which_tests == "32":
        ds = engine.get_32_gen()
    elif which_tests == "128":
        ds = engine.get_128_gen()
    elif which_tests == "of64":
        ds = engine.get_of64_gen()
    elif which_tests == "of32":
        ds = engine.get_of32_gen()
    elif which_tests == "of128":
        ds = engine.get_of128_gen()
    else:
        ds = engine.get_datasets()
    

    for d in ds:
        evaluate_model(results_folder, args, prep_gifs=prep_gifs, dist=d["dist"], thickness=d["thickness"], length=d["length"], which_tests=which_tests)


def evaluate_model(results_folder, args, prep_gifs=0, dist=14, thickness=1, length=64, which_tests="all"):
    """Evaluate a model and plot results."""
    os.makedirs(results_folder, exist_ok=True)
    model = engine.model_selector(args=args, timesteps=length, device=device)

    if "of" in which_tests:
        pf_root, timesteps, len_train_loader, len_val_loader = engine.of_dataset_selector(dist=dist, thickness=thickness, length=length)
    else: 
        pf_root, timesteps, len_train_loader, len_val_loader = engine.dataset_selector(dist=dist, thickness=thickness, length=length)
    print("Loading training dataset")
    train_loader = tfr_data_loader(data_dir=os.path.join(pf_root,'train-*'), batch_size=args.batch_size, drop_remainder=True, timesteps=timesteps)

    print("Loading validation dataset")
    val_loader = tfr_data_loader(data_dir=os.path.join(pf_root, 'test-*'), batch_size=args.batch_size, drop_remainder=True, timesteps=timesteps)


    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    # noqa Save timesteps/kernel_size/dimensions/learning rate/epochs/exp_name/algo/penalty to a dict for reloading in the future
    param_names_shapes = {k: v.shape for k, v in model.named_parameters()}
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    print("Including parameters {}".format([k for k, v in model.named_parameters()]))

    assert args.ckpt is not None, "You must pass a checkpoint for testing."
    model = engine.load_ckpt(model, args.ckpt, strict=args.strict)

    model.eval()
    accs = []
    losses = []
    for epoch in range(1):

        time_since_last = time.time()
        model.train()
        end = time.perf_counter()

        if debug_data:  # "skip" in pf_root:
            loader = train_loader
        else:
            loader = val_loader
        for idx, (imgs, target) in tqdm(enumerate(loader), total=int(len_val_loader / args.batch_size), desc="Processing test images"):

            # Get into pytorch format
            with torch.no_grad():
                imgs, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels)
                output, states, gates = engine.model_step(model, imgs, model_name=args.model, test=True)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target.float().reshape(-1, 1))
                accs.append((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean().cpu())
                losses.append(loss.item())
                if plot_incremental and "hgru" in args.model:
                    engine.plot_results(states, imgs, target, output=output, timesteps=timesteps, gates=gates)

    print("Dataset: Dist {} Thickness {} Len {}, Mean accuracy: {}, mean loss: {}".format(dist, thickness, length, np.mean(accs), np.mean(losses)))
    np.savez(os.path.join(results_folder, "test_perf_dist_{}_thickness_{}_length_{}".format(dist, thickness, length)), acc=np.mean(accs), loss=np.mean(losses), scores=output.reshape(-1).cpu())
    np.savez(os.path.join(results_folder, "exps_{}_dist_{}_thickness_{}_length_{}".format(which_tests, dist, thickness, length)), np.mean(accs), np.mean(losses))

    # Prep_gifs needs to be an integer
    if "hgru" in args.model and prep_gifs:
        data_results_folder = os.path.join(results_folder, "test_dist_{}_thickness_{}_length_{}".format(dist, thickness, length))
        os.makedirs(data_results_folder, exist_ok=True)
        engine.plot_results(states, imgs, target, output=output, timesteps=timesteps, gates=gates, prep_gifs=prep_gifs, results_folder=data_results_folder)

    if args.save_rnn:
        data_results_folder = os.path.join(results_folder, "recurrent_states_{}_thickness_{}_length_{}".format(dist, thickness, length))
        os.makedirs(data_results_folder, exist_ok=True)
        np.save(os.path.join(data_results_folder, "gates"), gates)
        np.save(os.path.join(data_results_folder, "states"), states)
        np.save(os.path.join(data_results_folder, "imgs"), imgs)


if __name__ == '__main__':
    length = args.length
    thickness = args.thickness
    dist = args.dist
    prep_gifs = 5  # args.gifs
    save_rnn = args.save_rnn
    which_tests = args.which_tests
    # perfs = np.load(os.path.join(directory, "val.npz"))["loss"]
    # arg_perf = np.argmin(perfs)
    if "of" in which_tests:
        res_dir = "_{}_{}_{}".format(length, thickness, dist)
        print("RUNNING OF MODEL")
    else:
        res_dir = "{}_{}_{}".format(length, thickness, dist)
    results_folder = os.path.join('results', res_dir, args.name)
    print("SAVING RESULTS TO {}".format(results_folder))
    if args.ckpt is None:
        eval_best_model(directory=results_folder, model=args.model, prep_gifs=prep_gifs, strict=args.not_strict, save_rnn=save_rnn, which_tests=which_tests)
    else:
        evaluate_model(results_folder=results_folder, args=args, prep_gifs=prep_gifs, save_rnn=save_rnn, which_tests=which_tests)


