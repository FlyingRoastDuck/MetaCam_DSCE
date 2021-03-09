import torch
import torch.nn.functional as F
from torch import nn, autograd


class EM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # # momentum update, not applied for meta learning
        # for x, y in zip(inputs, indexes):
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def em(inputs, indexes, features, momentum=0.5):
    return EM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class Memory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes, symmetric=False):
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        inputs = em(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()  # 16522, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        # re-arange simi matrix according to labels to find centroids
        sim.index_add_(0, labels[labels != -1], inputs[:, labels != -1].t().contiguous())

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        # get counter
        nums.index_add_(0, labels[labels != -1], torch.ones(labels[labels != -1].shape[0], 1).float().cuda())
        sim /= nums.clone().expand_as(sim)  # compute centroids
        softMask = torch.zeros(sim.t().shape).cuda()
        softMask.scatter_(1, targets.view(-1, 1), 1)
        loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
        loss_sym = 0
        if symmetric:
            loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask, dim=1)).sum(1).mean()
        return loss + loss_sym


class CamMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(CamMemory, self).__init__()
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features).to(self.devices))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long().to(self.devices))

        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638

        self.global_std, self.global_mean = torch.zeros(num_features).to(self.devices), \
                                            torch.zeros(num_features).to(self.devices)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def __update_params(self):
        camSet = set(self.cam.cpu().numpy().tolist())
        temp_std, temp_mean = [], []
        for cam in camSet:
            cam_feat = self.features[self.cam == cam]
            if len(cam_feat) <= 1: continue
            temp_std.append(cam_feat.std(0))
            temp_mean.append(cam_feat.mean(0))
        self.global_std = self.momentum * torch.stack(temp_std).mean(0) + \
                          (1 - self.momentum) * self.global_std
        self.global_mean = self.momentum * torch.stack(temp_mean).mean(0) + \
                           (1 - self.momentum) * self.global_mean

    def forward(self, features, indexes, cameras, symmetric=False):
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        self.__update_params()  # update camera-level params
        inputs = em(features, indexes, self.features, self.momentum)
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()  # 13638, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda()  # 12123(maxID)*64
        # re-arange simi matrix according to labels
        sim.index_add_(0, labels, inputs.t().contiguous())  # labels--13638(centroids+tgt IDs), inputs--13638*64

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()  # 12123(maxID)
        # get counter
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())

        sim /= nums.clone().expand_as(sim)

        # get camera loss
        num_cams, cam_set, loss_cam = len(set(self.cam)), set(self.cam.cpu().numpy().tolist()), []
        for cur_cam in range(len(cam_set)):
            cam_feat = features[cur_cam == cameras]
            if len(cam_feat) <= 1:
                continue
            temp_mean, temp_std = cam_feat.mean(0), cam_feat.std(0)

            loss_mean = (temp_mean - self.global_mean).pow(2).sum()
            loss_std = (temp_std - self.global_std).pow(2).sum()
            loss_cam.append(loss_mean)
            loss_cam.append(loss_std)

        softMask = torch.zeros(sim.t().shape).cuda()
        softMask.scatter_(1, targets.view(-1, 1), 1)
        loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
        loss_sym = 0
        if symmetric:
            loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask, dim=1)).sum(1).mean()

        loss_cam = 0 if len(loss_cam) == 0 else torch.stack(loss_cam).mean()

        return loss + loss_sym + loss_cam
