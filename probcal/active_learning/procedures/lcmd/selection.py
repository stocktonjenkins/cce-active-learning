from abc import ABC
import numpy as np
import torch
from .features import *


class SelectionMethod:
    """
    Abstract base class for selection methods,
    which allow to select a subset of indices from the pool set as the next batch to label for Batch Active Learning.
    """

    def __init__(self):
        super().__init__()
        self.status = None  # can be used to report errors during selection

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Select batch_size elements from the pool set
        (which is assumed to be given in the constructor of the corresponding subclass).
        This method needs to be implemented by subclasses.
        It is assumed that this method is only called once per object, since it can modify the state of the object.
        :param batch_size: Number of elements to select from the pool set.
        :return: Returns a torch.Tensor of integer type that contains the selected indices.
        """
        raise NotImplementedError()

    def get_status(self) -> Optional:
        """
        :return: Returns an object representing the status of the selection. If all went well, the method returns None.
        Otherwise, it might return a string or something different representing an error that occured.
        This is mainly useful for analyzing a lot of experiment runs.
        """
        return self.status


class IterativeSelectionMethod(SelectionMethod):
    """
    Abstract base class for iterative selection methods, as considered in the paper.
    """

    def __init__(
        self,
        pool_features: Features,
        train_features: Features,
        sel_with_train: bool,
        verbosity: int = 1,
    ):
        """
        :param pool_features: Features representing the pool set.
        :param train_features: Features representing the training set.
        :param sel_with_train: This corresponds to the mode parameter in the paper.
        Set to True if you want to use TP-mode (i.e. use the training data for selection), and to False for P-mode.
        :param verbosity: Level of verbosity. If >= 1, something may be printed to indicate the progress of selection.
        """
        super().__init__()
        self.train_features = train_features
        self.pool_features = pool_features
        self.features = (
            pool_features.concat_with(train_features)
            if sel_with_train
            else pool_features
        )
        self.selected_idxs = []
        self.selected_arr = torch.zeros(
            self.pool_features.get_n_samples(),
            dtype=torch.bool,
            device=self.pool_features.get_device(),
        )
        self.with_train = sel_with_train
        self.verbosity = verbosity
        self.n_added = 0

    def prepare(self, n_adds: int):
        """
        Callback method that may be implemented by subclasses.
        This method is called before starting the selection.
        It can be used, for example, to allocate memory depending on the batch size.
        :param n_adds: How often add() will be called during select()
        """
        pass

    def get_scores(self) -> torch.Tensor:
        """
        Abstract method that can be implemented by subclasses.
        This method should return a score for each pool sample,
        which can then be used to select the next pool sample to include.
        :return: Returns a torch.Tensor of shape [len(self.pool_features)] containing the scores.
        """
        raise NotImplementedError()

    def add(self, new_idx: int):
        """
        Update the state of the object (and therefore the scores) based on adding a new point to the selected set.
        :param new_idx: idx of the new point wrt to self.features,
        i.e. if new_idx > len(self.pool_features),
        then new_idx - len(self.pool_features) is the index for self.train_features,
        otherwise new_idx is an index to self.pool_features.
        """
        raise NotImplementedError()

    def get_next_idx(self) -> Optional[int]:
        """
        This method may be overridden by subclasses.
        It should return the index of the next sample that should be added to the batch.
        By default, it returns the index with the maximum score, according to self.get_scores().
        :return: Returns an int corresponding to the next index.
        It may also return None to indicate that the selection of the next index failed
        and that the batch should be filled up with random samples.
        """
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        return torch.argmax(self.get_scores()).item()

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Iterative implementation of batch selection for Batch Active Learning.
        :param batch_size: Number of elements that should be included in the batch.
        :return: Returns a torch.Tensor of integer type containing the indices of the selected pool set elements.
        """
        device = self.pool_features.get_device()

        self.prepare(
            batch_size + len(self.train_features) if self.with_train else batch_size
        )

        if self.with_train:
            # add training points first
            for i in range(len(self.train_features)):
                self.add(len(self.pool_features) + i)
                self.n_added += 1
                if (i + 1) % 256 == 0 and self.verbosity >= 1:
                    print(f"Added {i+1} train samples to selection", flush=True)

        for i in range(batch_size):
            next_idx = self.get_next_idx()
            if (
                next_idx is None
                or next_idx < 0
                or next_idx >= len(self.pool_features)
                or self.selected_arr[next_idx]
            ):
                # data selection failed
                # fill up with random remaining indices
                # print(f'{next_idx=}, {len(self.pool_features)=}')
                self.status = f"filling up with random samples because selection failed after n_selected = {len(self.selected_idxs)}"
                if self.verbosity >= 1:
                    print(self.status)
                n_missing = batch_size - len(self.selected_idxs)
                remaining_idxs = torch.nonzero(~self.selected_arr).squeeze(-1)
                new_random_idxs = remaining_idxs[
                    torch.randperm(len(remaining_idxs), device=device)[:n_missing]
                ]
                selected_idxs_tensor = torch.as_tensor(
                    self.selected_idxs, dtype=torch.long, device=torch.device(device)
                )
                return torch.cat([selected_idxs_tensor, new_random_idxs], dim=0)
            else:
                self.add(next_idx)
                self.n_added += 1
                self.selected_idxs.append(next_idx)
                self.selected_arr[next_idx] = True
        return torch.as_tensor(
            self.selected_idxs, dtype=torch.long, device=torch.device(device)
        )


class LargestClusterMaxDistSelectionMethod(IterativeSelectionMethod):
    """
    Implements the LCMD selection method for Batch Active Learning.
    """

    def __init__(
        self,
        pool_features: Features,
        train_features: Features,
        sel_with_train: bool = True,
        dist_weight_mode: str = "sq-dist",
    ):
        """
        :param pool_features:
        :param train_features:
        :param sel_with_train:
        :param dist_weight_mode: one of 'none', 'dist' or 'sq-dist'
        """
        super().__init__(
            pool_features=pool_features,
            train_features=train_features,
            sel_with_train=sel_with_train,
        )
        self.dist_weight_mode = dist_weight_mode
        self.min_sq_dists = np.Inf * torch.ones(
            self.pool_features.get_n_samples(),
            dtype=pool_features.get_dtype(),
            device=pool_features.get_device(),
        )
        self.closest_idxs = torch.zeros(
            self.pool_features.get_n_samples(),
            device=pool_features.get_device(),
            dtype=torch.long,
        )
        self.neg_inf_tensor = torch.as_tensor(
            [-np.Inf],
            dtype=pool_features.get_dtype(),
            device=pool_features.get_device(),
        )

    def get_scores(self) -> torch.Tensor:
        if self.dist_weight_mode == "sq-dist":
            weights = self.min_sq_dists
        elif self.dist_weight_mode == "dist":
            weights = self.min_sq_dists.sqrt()
        else:
            weights = None
        bincount = torch.bincount(
            self.closest_idxs, weights=weights, minlength=self.n_added + 1
        )
        max_bincount = torch.max(bincount)
        return torch.where(
            bincount[self.closest_idxs] == max_bincount,
            self.min_sq_dists,
            self.neg_inf_tensor,
        )

    def get_next_idx(self) -> Optional[int]:
        if self.n_added == 0:
            # no point added yet, take point with largest norm
            return torch.argmax(self.pool_features.get_kernel_matrix_diag()).item()
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        idx = torch.argmax(scores).item()
        return idx

    def add(self, new_idx: int):
        sq_dists = self.features[new_idx].get_sq_dists(self.pool_features).squeeze(0)
        new_min = sq_dists < self.min_sq_dists
        self.closest_idxs[new_min] = self.n_added + 1
        self.min_sq_dists[new_min] = sq_dists[new_min]
