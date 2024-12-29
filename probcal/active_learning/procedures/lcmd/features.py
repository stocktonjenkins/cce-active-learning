# Derived implementation from https://github.com/dholzmueller/bmdal_reg

from typing import Optional, Any

import torch

from probcal.active_learning.procedures.lcmd.feature_data import (
    FeatureData,
    Indexes,
    ConcatFeatureData,
)
from probcal.active_learning.procedures.lcmd.feature_maps import FeatureMap


class Features:
    """
    This class represents a combination of a feature map and feature data.
    Hence, it implicitly represents a (precomputed or non-precomputed) feature matrix.
    Whenever operations between to Features objects are used,
    it is assumed that both objects share the same feature map.
    Features objects can be transformed in various ways. Here is an example of a typical use-case:
    # initialize feature_map, train_feature_data, pool_feature_data
    train_features = Features(feature_map, train_feature_data)
    pool_features = Features(feature_map, pool_feature_data)
    tfm = train_features.posterior_tfm(sigma=0.1)
    post_train_features = tfm(train_features)
    post_pool_features = tfm(pool_features)
    post_pool_variances = post_pool_features.get_kernel_matrix_diag()
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        feature_data: FeatureData,
        diag: Optional[torch.Tensor] = None,
    ):
        """
        :param feature_map: Feature map.
        :param feature_data: Data that serves as input to the feature map.
        :param diag: Optional parameter representing the precomputed kernel matrix diagonal, i.e.,
        feature_map.get_kernel_matrix_diag(feature_data).
        """
        self.feature_map = feature_map
        self.feature_data = feature_data
        self.diag = diag
        if diag is not None and not isinstance(diag, torch.Tensor):
            raise ValueError(f"diag has wrong type {type(diag)}")

    def precompute(self) -> "Features":
        """
        :return: Returns a Features object where the feature map is precomputed on the feature data,
        i.e., some methods should be faster to evaluate on the precomputed Features object.
        """
        fm, fd = self.feature_map.precompute(self.feature_data)
        if self.diag is None:
            self.diag = fm.get_kernel_matrix_diag(fd)
        return Features(fm, fd, self.diag)

    def simplify(self) -> "Features":
        """
        :return: Returns a Features object where the feature data is simplified (un-batched etc.),
        which potentially makes evaluations faster, similar to precompute().
        """
        return Features(self.feature_map, self.feature_data.simplify(), self.diag)

    def get_n_samples(self) -> int:
        """
        :return: Returns the number of samples in self.feature_data.
        """
        return self.feature_data.get_n_samples()

    def __len__(self) -> int:
        """
        :return: Returns the number of samples in self.feature_data.
        """
        return self.get_n_samples()

    def get_n_features(self) -> int:
        """
        :return: Returns the number of features of the corresponding feature map.
        """
        return self.feature_map.get_n_features()

    def get_device(self) -> str:
        """
        :return: Returns the (torch) device that the feature data is on.
        """
        return self.feature_data.get_device()

    def get_dtype(self) -> Any:
        """
        :return: Returns the (torch) dtype that the feature data has.
        """
        return self.feature_data.get_dtype()

    def __getitem__(self, idxs: int | slice | torch.Tensor) -> "Features":
        """
        Returns a Features object where the feature data is indexed by idxs. Note that if idxs is an integer,
        the feature data will be indexed by [idxs:idxs+1], i.e. the batch dimension will not be removed
        and the resulting feature data tensors will have shape [1, ...].
        This method is called when using an indexing expression
        such as features[0] or features[-2:] or features[torch.Tensor([0, 2, 4], dtype=torch.long)]
        :param idxs: Integer, slice, or torch.Tensor of integer type.
        See the comment above about indexing with integers.
        :return: Returns a Feature object where the feature data is the indexed version of self.feature_data.
        """
        idxs = Indexes(self.get_n_samples(), idxs)
        return Features(
            self.feature_map,
            self.feature_data[idxs],
            None if self.diag is None else self.diag[idxs.get_idxs()],
        )

    def get_kernel_matrix_diag(self) -> torch.Tensor:
        """
        Returns the kernel matrix diagonal
        obtained by self.feature_map.get_kernel_matrix_diag(self.feature_data).
        The kernel matrix diagonal is stored and reused,
        such that multiple calls to this method do not trigger multiple computations.
        Consequently, the returned Tensor should not be modified.
        :return: Returns a torch.Tensor of shape [len(self)] containing the kernel matrix diagonal.
        """
        if self.diag is None:
            self.diag = self.feature_map.get_kernel_matrix_diag(self.feature_data)
        return self.diag

    def get_kernel_matrix(self, other_features: "Features") -> torch.Tensor:
        """
        Returns the kernel matrix k(self.feature_data, other_features.feature_data),
        where k is the kernel given by self.feature_map.
        :param other_features: Other features corresponding to the columns of the resulting kernel matrix.
        :return: Returns a torch.Tensor of shape [len(self), len(other_features)] corresponding to the kernel matrix.
        """
        return self.feature_map.get_kernel_matrix(
            self.feature_data, other_features.feature_data
        )

    def get_feature_matrix(self) -> torch.Tensor:
        """
        :return: Returns self.feature_map.get_feature_matrix(self.feature_data), i.e.,
        a torch.Tensor of shape [len(self), self.get_n_features()] containing the feature matrix.
        Note that this method cannot be used for all feature maps,
        since they might have an infinite-dimensional feature space.
        """
        return self.feature_map.get_feature_matrix(self.feature_data)

    def get_sq_dists(self, other_features: "Features") -> torch.Tensor:
        """
        Return a matrix containing the squared feature space distances
        between self.feature_data and other_features.feature_data. These are computed using the kernel,
        hence they also work for feature maps with infinite-dimensional feature space.
        :param other_features: Features object to compute the distances to.
        :return: Returns a torch.Tensor of shape [len(self), len(other_features)]
        representing the squared distances between the feature data in feature space.
        """
        diag = self.get_kernel_matrix_diag()
        other_diag = other_features.get_kernel_matrix_diag()
        kernel_matrix = self.get_kernel_matrix(other_features)
        sq_dists = diag[:, None] + other_diag[None, :] - 2 * kernel_matrix
        return sq_dists

    def concat_with(self, other_features: "Features"):
        """
        Concatenates two features objects along the sample dimension.
        :param other_features: Other Features object to concatenate with self.
        :return: Returns a Features object where self.feature_data and other_features.feature_data
        are concatenated along the batch dimension.
        """
        diag = (
            torch.cat([self.diag, other_features.diag], dim=0)
            if self.diag is not None and other_features.diag is not None
            else None
        )
        return Features(
            self.feature_map,
            ConcatFeatureData([self.feature_data, other_features.feature_data]),
            diag,
        )


class FeaturesTransform:
    """
    Abstract base class for classes that allow to transform a Features object into another Features object
    """

    def __call__(self, features: Features) -> Features:
        """
        This method should be overridden by subclasses.
        :param features: Features object to transform.
        :return: Returns the transformed Features object.
        """
        raise NotImplementedError()
