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
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8


init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 32
weight_decay = 2e-4
num_workers = 4
T = 2


class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.loadpre = args["loadpre"]
        self.method = args["method"]
        self.increment = args["increment"]
        self.dataset = args["dataset"]
    def after_task(self):
        print(f"batchsize is {batch_size}")
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

        # Loader
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

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
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
        if self.loadpre == 1:
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

            # model_save_path = f"replay_imagenet_model_seed1993_half.pth"

            # torch.save(self._network.state_dict(), model_save_path)
            logging.info(info)
        else:
            model_path = ""
            if self.dataset == "cifar100":
                model_path = f"icarl_model_seed1993_half.pth"
            else:
                model_path = f"replay_imagenet_model_seed1993_half.pth"

            print(model_path)
            self._network.load_state_dict(torch.load(model_path))
            print(f"change_lwf_loading~~~~")
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = 0

                if self.method == "normal":
                    loss_kd = 0
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
                    loss_kd = loss_kd1 + 1/2 * loss_kd2
                loss =  (loss_kd) + loss_clf
                # loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
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
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


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
