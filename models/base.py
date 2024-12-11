import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
import PIL

from copy import deepcopy
from typing import List, Tuple
from torchvision.transforms import functional as TF
from torchvision import transforms

EPSILON = 1e-8
batch_size = 64




class BaseSampleSelection:
    """
    Base class for sample selection strategies.
    """

    def __init__(self, buffer_size: int, device):
        """
        Initialize the sample selection strategy.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the buffer on
        """
        self.buffer_size = buffer_size
        self.device = device

    def __call__(self, num_seen_examples: int) -> int:
        """
        Selects the index of the sample to replace.

        Args:
            num_seen_examples: the number of seen examples

        Returns:
            the index of the sample to replace
        """

        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        (optional) Update the state of the sample selection strategy.
        """
        pass

class ReservoirSampling(BaseSampleSelection):
    def __call__(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < 2000:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples)
        if rand < 2000:
            return rand
        else:
            return -1



class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.increment = args["increment"]
#################
        self.attributes = ['examples',  'logits','labels', 'task_labels', 'true_labels']
        self.sample_selection_fn = ReservoirSampling(self._memory_size, self._device)
        self.num_seen_examples = len(self._data_memory)
        self.buffer_size = 128 
        self.device = self._device
###################       
    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        self.sample_selection_fn.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self
##################

    def apply_transform(self , x: torch.Tensor, transform, autosqueeze=False) -> torch.Tensor:
        """Applies a transform to a batch of images.

        If the transforms is a KorniaAugNoGrad, it is applied directly to the batch.
        Otherwise, it is applied to each image in the batch.

        Args:
            x: a batch of images.
            transform: the transform to apply.
            autosqueeze: whether to automatically squeeze the output tensor.

        Returns:
            The transformed batch of images.
        """
        if transform is None:
            return x

        # if isinstance(x, PIL.Image.Image):
            # print(f"isintstance is ssxxxxxxtan{transform}")
        return transform(x)
        
        # print(f"isintstance is ssxxxxxxtan{transform}")
        out = torch.stack([xi for xi in x.cpu()], dim=0).to(self._device)
        if autosqueeze and out.shape[0] == 1:
            out = out.squeeze(0)
        return out

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     true_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: tensor containing the true labels (used only for logging)
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self._memory_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

#################################

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None, true_labels=None, sample_selection_scores=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            attention_maps: list of tensors containing the attention maps
            true_labels: if setting is noisy, the true labels associated with the examples. **Used only for logging.**
            sample_selection_scores: tensor containing the scores used for the sample selection strategy. NOTE: this is only used if the sample selection strategy defines the `update` method.

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):
            # print(f"hererererer")
            self.init_tensors(examples, labels, logits, task_labels, true_labels)

        for i in range(examples.shape[0]):
            
            index = self.sample_selection_fn(self.num_seen_examples)

            # print(f" {i}st examples change index is { index }")
            self.num_seen_examples += 1
            if index >= 0:
                # print(f"self.examples {self.examples}")
                # print(f"examples is ok,index is {index}")
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    # print(f"line 203")
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    # print(f"logit ok , index is {index} , self.logits[index] is {self.logits[index]}")
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if attention_maps is not None:
                    self.attention_maps[index] = [at[i].byte().to(self.device) for at in attention_maps]
                if sample_selection_scores is not None:
                    self.sample_selection_fn.update(index, sample_selection_scores[i])
                if true_labels is not None:
                    self.true_labels[index] = true_labels[i].to(self.device)
####################################

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None,
                 mask_task_out=None, cpt=None, return_not_aug=False, not_aug_transform=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)
            return_not_aug: if True, also returns the not augmented items
            not_aug_transform: the transformation to be applied to the not augmented items (if `return_not_aug` is True)

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device = self.device if device is None else device

        num_avail_samples = self.examples.shape[0] if mask_task_out is None else samples_mask.sum().item()
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)

        if size > min(num_avail_samples, self.examples.shape[0]):
            size = min(num_avail_samples, self.examples.shape[0])
        # print(f"line 238")
        choice = np.random.choice(num_avail_samples, size=size, replace=True)
        # print(f"line 240")
        # MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        # TRANSFORM = transforms.Compose(
        #     [transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(MEAN, STD)])
        # transform = transforms.Compose(
        #     [transforms.ToPILImage(), TRANSFORM])
        if transform is None:
            def transform(x): return x

        # print(f"line 252")
        selected_samples = self.examples[choice] 
        # print(f"line 254")
        ret_tuple = tuple()
        
        ret_tuple += (self.apply_transform(selected_samples, transform=transform).to(target_device),)
        # print(f"line 258")
        # print(f"ret_tuple is {ret_tuple}")
        for attr_str in self.attributes[1:]:
            # print(f"line 261 ,attr_str is {attr_str}")
            if hasattr(self, attr_str):
                # print(f"attr_str is {attr_str}")
                # print(f"choice is {choice}")
                # print(f"logits is {self.logits.shape}")

                attr = getattr(self, attr_str)
                # print(f"attr_str is {attr_str}, attr is {attr}, choice is {choice}, mask_task_out is {mask_task_out}")
                selected_attr = attr[choice] 
                ret_tuple += (selected_attr.to(target_device),)
        # print(f"line 267")
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple    

#####################################

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False
######################################

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.increment)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            print(f"11111111111111111111111")
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            print(f"22222222222222222222222222222222")
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self.device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self.device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
