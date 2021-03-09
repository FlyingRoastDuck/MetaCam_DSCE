from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F

from reid import models
from reid.models.em import Memory
from reid.trainers import Trainer_USL
from reid.evaluators import Evaluator, extract_features
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.faiss_rerank import compute_jaccard_distance
from reid.utils import generate_pseudo_labels
from reid import datasets
from reid.utils.tools import get_test_loader, get_plot_loader, get_train_loader
from reid.utils.tsne import plotTSNE
from scipy import io

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    model = model.cuda()
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.copy_weight(checkpoint['state_dict'])
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        return

    # # for vis
    # marCamSet = get_data('marCam', args.data_dir)
    # mar_loader = get_plot_loader(marCamSet, args.height, args.width,
    #                              args.batch_size, args.workers, test_set=marCamSet.train)

    # Create feature memory
    memory = nn.DataParallel(
        Memory(2048, len(dataset.train),
               temp=args.temp, momentum=args.momentum)
    ).cuda()

    # Initialize target-domain instance features
    print("==> Initialize instance features in the feature memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory.module.features = F.normalize(features, dim=1).cuda()

    del cluster_loader

    # optimizer for meta models
    params = [{"params": [value]} for value in model.module.params() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer_USL(model, memory)
    cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=-1)
    # instance pre-training
    pseudo_labeled_dataset = []
    pseudo_labels = torch.arange(len(dataset.train))
    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
        pseudo_labeled_dataset.append((fname, label.item(), cid))
    for epoch in range(args.startE):
        torch.cuda.empty_cache()
        memory.module.labels = pseudo_labels.cuda()
        train_loader = get_train_loader(dataset.images_dir, args.height, args.width,
                                        args.batch_size, args.workers, -1, iters,
                                        trainset=pseudo_labeled_dataset)
        print(f'-----Exemplar Pretraining, Epoch{epoch}...------')
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters)
    # test pre-train
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
    # start training
    for epoch in range(args.epochs):
        # Calculate distance
        torch.cuda.empty_cache()
        features = memory.module.features.clone()
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)

        # generate new dataset and calculate cluster centers
        pseudo_labels = generate_pseudo_labels(pseudo_labels, features)
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pseudo_labeled_dataset.append((fname, label.item(), cid))
        # statistics of clusters and un-clustered instances
        memory.module.labels = pseudo_labels.cuda()
        train_loader = get_train_loader(dataset.images_dir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters, symmetric=args.symmetric)
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            # # for vis
            # mar_feature, _ = extract_features(model, mar_loader, print_freq=args.print_freq)
            # mar_feature = torch.stack([mar_feature[f] for f, _, _ in marCamSet.train], 0)
            # marPid, marCam = [pid for _, pid, _ in marCamSet.train], \
            #                  [cam for _, _, cam in marCamSet.train]
            # tsneCam = plotTSNE(mar_feature, marPid, marCam, f'{epoch}.jpg')
            # io.savemat(f'{epoch}.mat', {'tsneCam': tsneCam, 'marPid': marPid, 'marCam': marCam})

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaCam with ACT Merge")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resMeta',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the feature memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--startE', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--symmetric', action='store_true',
                        help="for sym ce")
    main()
