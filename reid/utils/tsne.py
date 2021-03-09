import matplotlib as mpl

mpl.use("Agg")
import pylab as pl
import seaborn as sns

sns.set()

from sklearn.manifold import TSNE
from ..evaluators import extract_features
import torch
import os


def plotTSNE(features, pids, cams, savePath):
    func = TSNE()
    embFeat = func.fit_transform(features)
    pl.figure()
    sns.scatterplot(embFeat[:, 0], embFeat[:, 1], hue=pids,
                    style=cams, legend=False, palette="nipy_spectral")
    pl.savefig(savePath)
    pl.close()
    return embFeat


def showTSNE(model, camLoader, camSet, epoch, dName):
    marCamFeat, marIDs = extract_features(model, camLoader, print_freq=50)
    marCamIDs = torch.tensor([camID for _, _, camID in sorted(camSet.train)]).numpy()
    marCamFeat = torch.cat([marCamFeat[f].unsqueeze(0) for f, _, _ in sorted(camSet.train)], 0)
    marIDs = torch.tensor([pid for _, pid, _ in sorted(camSet.train)]).numpy()
    saveFolder = os.path.join(os.getcwd(), f'{dName}_tsneFolder')
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    plotTSNE(marCamFeat, marIDs, marCamIDs, os.path.join(saveFolder, f"{epoch}.jpg"))
