import gc

import numpy as np
import torch
from tqdm import tqdm

from probcal.active_learning.active_learning_types import (
    ActiveLearningEvaluationResults,
)
from probcal.active_learning.procedures.base import ActiveLearningProcedure
from probcal.models.regression_nn import RegressionNN


class BAITProcedure(ActiveLearningProcedure[ActiveLearningEvaluationResults]):
    def __init__(
        self,
        dataset,
        config,
        fisher_batch_size=1000,
        lamb=1,
        device="cpu",
    ):
        super().__init__(dataset, config)
        self.fisher_batch_size = fisher_batch_size
        self.device = device
        self.lamb = lamb

    def get_next_label_set(
        self,
        unlabeled_indices: np.ndarray,
        k: int,
        model: RegressionNN,
    ) -> np.ndarray:
        """
        Choose the next set of indices to add to the label set based on Fisher Information.

        Args:
            model: RegressionNN
            unlabeled_indices: np.ndarray
            k: int

        Returns:
            A subset of unlabeled indices selected based on Fisher Information.
        """
        train_dataloader = self.dataset.train_dataloader()
        unlabeled_dataloader = self.dataset.unlabeled_dataloader()

        # get low-rank point-wise fishers
        xt_unlabeled = self.get_embedding(model, unlabeled_dataloader).unsqueeze(1)
        xt_labeled = self.get_embedding(model, train_dataloader).unsqueeze(1)
        xt = torch.cat([xt_unlabeled, xt_labeled], dim=0)

        # get fisher
        print("getting fisher matrix...", flush=True)
        batchSize = self.fisher_batch_size  # should be as large as gpu memory allows
        # nClass = torch.max(self.Y).item() + 1
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        for i in range(int(np.ceil(xt.shape[0] / batchSize))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
            op = (
                torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0)
                .detach()
                .cpu()
            )
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        # nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt_labeled
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
            op = (
                torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0)
                .detach()
                .cpu()
            )
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = self.select(
            xt_unlabeled, k, fisher, init, lamb=self.lamb, nLabeled=xt_labeled.shape[0]
        )
        return unlabeled_indices[chosen]

    def select(self, X, K, fisher, iterates, lamb=1, nLabeled=0):
        indsAll = []
        dim = X.shape[-1]
        rank = X.shape[-2]

        currentInv = torch.inverse(
            lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K)
        )
        X = X * np.sqrt(K / (nLabeled + K))
        fisher = fisher.cuda()

        # forward selection, over-sample by 2x
        print("forward selection...", flush=True)
        over_sample = 2
        for i in tqdm(range(int(over_sample * K))):
            # check trace with low-rank updates (woodbury identity)
            xt_ = X.cuda()
            innerInv = torch.inverse(
                torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            innerInv[torch.where(torch.isinf(innerInv))] = (
                torch.sign(innerInv[torch.where(torch.isinf(innerInv))])
                * np.finfo("float32").max
            )
            traceEst = torch.diagonal(
                xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
                dim1=-2,
                dim2=-1,
            ).sum(-1)

            # clear out gpu memory
            xt = xt_.cpu()
            del xt, innerInv
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            # get the smallest unselected item
            traceEst = traceEst.detach().cpu().numpy()
            for j in np.argsort(traceEst)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            # print(i, ind, traceEst[ind], flush=True)

            # commit to a low-rank update
            xt_ = X[ind].unsqueeze(0).cuda()
            innerInv = torch.inverse(
                torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            currentInv = (
                currentInv
                - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
            ).detach()[0]

        # backward pruning
        print("backward pruning...", flush=True)
        for i in tqdm(range(len(indsAll) - K)):
            # select index for removal
            xt_ = X[indsAll].cuda()
            innerInv = torch.inverse(
                -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            traceEst = torch.diagonal(
                xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
                dim1=-2,
                dim2=-1,
            ).sum(-1)
            delInd = torch.argmin(-1 * traceEst).item()
            # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

            # low-rank update (woodbury identity)
            xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
            innerInv = torch.inverse(
                -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            currentInv = (
                currentInv
                - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
            ).detach()[0]

            del indsAll[delInd]

        del xt_, innerInv, currentInv
        torch.cuda.empty_cache()
        gc.collect()
        return indsAll
