import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import pandas as pd
import os
import math

init_epoch = 120
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 80
lrate = 0.1
milestones = [60, 120, 180, 220]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 1
lambda_f_base = 1




class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._final_task_id = (100-args["init_cls"])/args["increment"]
        self.class_weight_norms = []  # 存储每个类权重范数的列表
        self.class_task1_start = []
        
        self.method = args["method"]
        self.increment = args["increment"]
        self.loadpre = args["loadpre"]
        self.init_cls = args['init_cls']
        self.increment = args["increment"]
        self.lambd    = args["lambd"]
        self.alpha    = args["alpha"]
        self.transform = []
        if self.lambd == -1:
            self.gala = self.alpha
            self.beta  = 0
        else:
            self.gala = self.alpha * self.lambd / (1+self.lambd)
            self.beta  = self.alpha * 1.0 / (1+self.lambd)
        self.model_path = args["path"]
        self._network.update_fc(100)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        print(f"self._known_classes is {self._known_classes},self.total_classes is {self._total_classes}")
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        # print(train_dataset.images[0])

        self.transform = train_dataset.trsf

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers ,pin_memory=True
        )
        print(f"cur_task is {self._cur_task} , the total_classes is {self._total_classes} , finel_task_id is {self._final_task_id} , initclass num is {data_manager.get_task_size(0)}")
        test_data, test_Label,test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",ret_data = True
        )


        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,
        )


        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader ):
        print(f" gala is {self.gala} , beta is {self.beta}")
        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(
                self._total_classes / (self._total_classes - self._known_classes)
            )
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        optimizer = optim.SGD(
            self._network.parameters(),
            lr=lrate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=lrate_decay
        )
        self._update_representation(train_loader, test_loader,optimizer, scheduler)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        # if self._cur_task == 0 and self.loadpre == 0:
        #     print(f"model_path is {self.model_path}")
        #     self._network.load_state_dict(torch.load(self.model_path))
        # else:
        prog_bar = tqdm(range(epochs))
        times = 0

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            clfloss = 0.0
            kd_losses = 0.0
            dark_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets ) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                # old_logits = self._old_network(inputs)["logits"]

                # fake_targets = targets - self._known_classes
                # loss_clf = F.cross_entropy(
                #     logits[:, self._known_classes:], fake_targets
                # )    
                loss_clf = F.cross_entropy(
                    logits, targets
                )
                loss_clf.backward()    
                loss = loss_clf.item()
                clfloss += loss_clf.item()
                if self.method == "der":
                    if not self.is_empty():
                        buf_inputs, buf_logits = self.get_data(batch_size, device = self._device)
                        buf_outputs = self._network(buf_inputs)["logits"]

                        loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
                        loss_mse.backward()

                        kd_losses += loss_mse.item()
                        loss += loss_mse.item()  

                elif self.method == "der_interintra":
                    if not self.is_empty():
                        buf_inputs, buf_logits = self.get_data(batch_size, device = self._device)

                        buf_outputs = self._network(buf_inputs)["logits"]

                        Zscore_logits = Zscore(buf_outputs[:, :self._total_classes])
                        Zscore_old = Zscore(buf_logits[:,:self._total_classes])

                        # Zscore_logits = Zscore(buf_outputs)
                        # Zscore_old = Zscore(buf_logits)

                        # print ( f' cur outputs shape is {Zscore_logits.shape}')

                        loss_kd1 =  DistillKL_logit_stand(
                        Zscore_logits[:, :self._total_classe],
                        Zscore_old[:, :self._total_classe],
                        T,
                        )

                        Zscore_dim1_logits = Inverse_Zscore(buf_outputs[:, :self._total_classes])
                        Zscore_dim1_old = Inverse_Zscore(buf_logits[:,:self._total_classes])                        

                        # Zscore_dim1_logits = Inverse_Zscore(buf_outputs)
                        # Zscore_dim1_old = Inverse_Zscore(buf_logits)  


                        loss_kd2 =  DistillKL_logit_stand(
                            Zscore_dim1_logits[:, :self._total_classe].t(),
                            Zscore_dim1_old[:, :self._total_classe].t(),
                            T,                            
                        )

                        loss_kd = self.gala * loss_kd1 + self.beta * loss_kd2

                        loss_kd.backward()
                        kd_losses += loss_kd.item()
                        loss += loss_kd.item()                      

                elif self.method == 'derpp':
                    if not self.is_empty():
                        buf_inputs, buf_logits,_ = self.get_data(batch_size, device = self._device)

                        buf_outputs = self._network(buf_inputs)["logits"]
                        loss_mse =  F.mse_loss(buf_outputs, buf_logits)
                        loss_mse.backward()

                        loss+= loss_mse.item()

                        kd_losses += loss_mse.item()

                        buf_inputs, _, buf_labels = self.get_data(batch_size, device=self._device)

                        buf_outputs = self._network(buf_inputs)["logits"]
                        loss_ce = 0.5* F.cross_entropy(buf_outputs, buf_labels)                       

                        loss_ce.backward()

                        loss += loss_ce.item()
                        dark_losses += loss_ce.item()

                elif self.method == 'derpp_interintra':
                    
                    if not self.is_empty():
                        buf_inputs,buf_logits,_  = self.get_data(batch_size, device = self._device)

                        buf_outputs = self._network(buf_inputs)["logits"]

                        # Zscore_logits = Zscore(buf_outputs[:, :self._total_classes])
                        # Zscore_old = Zscore(buf_logits[:,:self._total_classes])
                        
                        Zscore_logits = Zscore(buf_outputs)
                        Zscore_old = Zscore(buf_logits)

                        # print ( f' cur outputs shape is {Zscore_logits.shape}')

                        loss_kd1 = DistillKL_logit_stand(
                        Zscore_logits[:,:self._total_classes],
                        Zscore_old[:,:self._total_classes],
                        T,
                        )

                        # Zscore_dim1_logits = Inverse_Zscore(buf_outputs[:, :self._total_classes])
                        # Zscore_dim1_old = Inverse_Zscore(buf_logits[:,:self._total_classes])                        

                        Zscore_dim1_logits = Inverse_Zscore(buf_outputs)
                        Zscore_dim1_old = Inverse_Zscore(buf_logits)    

                        loss_kd2 = DistillKL_logit_stand(
                            Zscore_dim1_logits[:,:self._total_classes].t(),
                            Zscore_dim1_old[:,:self._total_classes].t(),
                            T,                            
                        )

                        loss_kd = self.gala * loss_kd1 + self.beta * loss_kd2
                        
                        loss_kd.backward()

                        loss+= loss_kd.item()

                        kd_losses += loss_kd.item()

                        buf_inputs, _,buf_labels = self.get_data(batch_size, device=self._device)

                        buf_outputs = self._network(buf_inputs)["logits"]
                        loss_ce = 0.5 * F.cross_entropy(buf_outputs, buf_labels)                       

                        loss_ce.backward()

                        loss += loss_ce.item()
                        dark_losses += loss_ce.item()
                optimizer.step()
                if self.method != 'derpp' and self.method != 'derpp_interintra' :
                    self.add_data(examples=inputs, logits=logits.data)  

                else:
                    self.add_data(examples=inputs, labels=targets, logits=logits.data )

                times += 1

                losses += loss
                

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)


            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f},Kd {:.3f},Clf {:.3f},Dark {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    clfloss / len(train_loader),
                    dark_losses / len(train_loader),
                    train_acc,
                    test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f},Kd {:.3f},Clf {:.3f},Dark {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    clfloss / len(train_loader),
                    dark_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        print(f"times is  {times}")
        logging.info(info)

            # if self._cur_task == 0 and self.loadpre == 1:
            #     torch.save(self._network.state_dict(), self.model_path)
            #     logging.info(info)


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

