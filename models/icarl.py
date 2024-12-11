import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import math

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005
lambda_f_base = 1


epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.method = args["method"]
        self.increment = args["increment"]
        self.dataset = args["dataset"]
        self.loadpre = args["loadpre"]
        self.init_cls = args["init_cls"]
        self.lambd = args["lambd"]

        if self.lambd == -1:
            self.alpha = 2
            self.beta  = 0
        else:
            self.alpha = 2.0 * self.lambd / (1+self.lambd)
            self.beta  = 2.0 * 1.0 / (1+self.lambd)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        print(f" alpha is {self.alpha} , beta is {self.beta}")
        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(
                self._total_classes / (self._total_classes - self._known_classes)
            )
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.loadpre == 1 :
            prog_bar = tqdm(range(init_epoch))
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]

                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        init_epoch,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        init_epoch,
                        losses / len(train_loader),
                        train_acc,
                    )

                prog_bar.set_description(info)

            # model_save_path = f"imagenet_model_seed1993_half.pth"
            # torch.save(self._network.state_dict(), model_save_path)
            logging.info(info)
        else:
            model_path = ''
            if self.init_cls == 50:
                model_path = f"icarl_model_seed1993_half.pth"
            elif self.init_cls == 20:
                model_path = f"model_seed1993_T5.pth"
            else:
                model_path = f"model_seed1993_T10.pth"

            # if self.init_cls == 50:
            #     model_path = f"imagenet_model_seed1993_half.pth"
            # elif self.init_cls == 20:
            #     model_path = f"imagenet_model_seed1993_T5.pth"
            # else:
            #     model_path = f"imagenet_model_seed1993_T10.pth"                
            print(model_path)
            self._network.load_state_dict(torch.load(model_path))
            print(f"change_lwf_loading~~~~")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))

        if self.method == "rkd":
            dist_criterion = RkdDistance()
            angle_criterion = RKdAngle()

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            kd_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)

                loss_kd = 0

                if self.method == "normal":
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs)["logits"],
                        T,
                    )
                elif self.method == "inter":
                    Zscore_logits = Zscore(logits)
                    Zscore_old = Zscore(self._old_network(inputs)["logits"])

                    loss_kd = DistillKL_logit_stand(
                        Zscore_logits[:, : self._known_classes],
                        Zscore_old,
                        T,
                    )
                elif self.method == "intra":
                    Zscore_dim1_logits = Inverse_Zscore(logits)
                    Zscore_dim1_old = Inverse_Zscore(self._old_network(inputs)["logits"])
                    loss_kd = DistillKL_logit_stand(
                        (Zscore_dim1_logits[:, : self._known_classes]).t(),
                        Zscore_dim1_old.t(),
                        T,
                    )
                elif self.method == "interintra":
                    Zscore_logits = Zscore(logits)
                    Zscore_old = Zscore(self._old_network(inputs)["logits"])

                    loss_kd1 = DistillKL_logit_stand(
                        Zscore_logits[:, : self._known_classes],
                        Zscore_old,
                        T,
                    )
                    Zscore_dim1_logits = Inverse_Zscore(logits)
                    Zscore_dim1_old = Inverse_Zscore(self._old_network(inputs)["logits"])
                    loss_kd2 = DistillKL_logit_stand(
                        (Zscore_dim1_logits[:, : self._known_classes]).t(),
                        Zscore_dim1_old.t(),
                        T,
                    )

                    loss_kd = self.alpha * loss_kd1 + self.beta * loss_kd2

                elif self.method == "feature":
                    old_features = self._old_network.get_features(inputs).detach()
                    features     = self._network.get_features(inputs)
                    flat_loss = (F.cosine_embedding_loss(
                                    features,
                                    old_features,
                                    torch.ones(inputs.shape[0]).to(self._device),
                                )
                                * self.factor
                                * lambda_f_base)
                            
                    loss_kd = 1 * flat_loss          
                elif self.method == "passfeature":
                    # features = self._network_module_ptr.extract_vector(inputs)
                    # features_old = self.old_network_module_ptr.extract_vector(inputs)
                    features_old = self._old_network.get_features(inputs).detach()
                    features     = self._network.get_features(inputs)                    
                    loss_kd =  5 * torch.dist(features, features_old, 2) 
                    
                elif self.method == "rkd":
                    features = self._network.get_features(inputs)
                    t_features = self._old_network.get_features(inputs).detach()

                    dist_loss = 1 * dist_criterion(features, t_features)
                    angle_loss = 2 * angle_criterion(features, t_features)

                    loss_kd = self.lambd * (dist_loss + angle_loss)      


                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                kd_losses += loss_kd.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f},Kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                    test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f},Kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

def Zscore(logits):
    mean = logits.mean(dim=-1, keepdims=True)
    stdv = logits.std(dim=-1, keepdims=True)
    return (logits - mean) / (1e-7 + stdv)

def Inverse_Zscore(logits):
    mean = logits.mean(dim=0, keepdims=True)
    stdv = logits.std(dim=0, keepdims=True)
    return (logits - mean) / (1e-7 + stdv)

def DistillKL_logit_stand(y_s,y_t,temp):
    """Distilling the Knowledge in a Neural Network"""
    T = temp
    KD_loss = 0
    KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s / T, dim=1),
                                                       F.softmax(y_t / T, dim=1)) * T * T
    return KD_loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        # print(f"teacher is {teacher.shape} , student is {student.shape}")
        
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)

            # print(f"t_d is {t_d.shape}")
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss