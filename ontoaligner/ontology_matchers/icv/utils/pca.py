# -*- coding: utf-8 -*-
"""
This script implements a Principal Component Analysis (PCA) algorithm using PyTorch.
The `PCA` class provides methods for fitting the model to data, transforming data into principal components,
and performing inverse transformations. Additionally, the script includes a helper function `svd_flip`
to ensure consistent signs for the singular value decomposition (SVD) components.

Classes:
    - PCA: A PyTorch module implementing Principal Component Analysis with methods for fitting, transforming, and inverse transforming data.

Functions:
    - svd_flip: Flips the singular vectors based on the largest element in each column of u to maintain consistency in signs.
"""

import torch
import torch.nn as nn


def svd_flip(u, v):
    """
    Flips the singular vectors to ensure consistency of signs based on the largest element in each column of `u`.

    Args:
        u (torch.Tensor): A tensor representing the left singular vectors of a matrix from SVD.
        v (torch.Tensor): A tensor representing the right singular vectors of a matrix from SVD.

    Returns:
        tuple: A tuple containing the flipped `u` and `v` tensors, ensuring consistent signs.
    """
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    """
    A PyTorch module that implements Principal Component Analysis (PCA).

    Args:
        n_components (int, optional): The number of principal components to keep. If None, keeps all components.
    """

    def __init__(self, n_components):
        """
        Initializes the PCA model.

        Args:
            n_components (int): The number of principal components to keep.
        """
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        """
        Fits the PCA model to the input data `X`.

        Args:
            X (torch.Tensor): The input data, where each row is a data sample.

        Returns:
            PCA: The PCA instance with learned components.
        """
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))  # Compute mean for centering
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)  # Perform SVD
        Vt = Vh
        U, Vt = svd_flip(U, Vt)  # Flip signs for consistency
        self.register_buffer("components_", Vt[:d])  # Store components
        return self

    def forward(self, X):
        """
        Transforms the input data `X` into the principal components.

        Args:
            X (torch.Tensor): The input data to transform.

        Returns:
            torch.Tensor: The transformed data in the principal component space.
        """
        return self.transform(X)

    def transform(self, X):
        """
        Projects the input data `X` onto the learned principal components.

        Args:
            X (torch.Tensor): The input data to project.

        Returns:
            torch.Tensor: The projected data in the principal component space.

        Raises:
            AssertionError: If the PCA model has not been fit before calling this method.
        """
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        """
        Fits the PCA model and then transforms the input data `X`.

        Args:
            X (torch.Tensor): The input data to fit and transform.

        Returns:
            torch.Tensor: The transformed data in the principal component space.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        """
        Reconstructs the original data from the transformed data `Y`.

        Args:
            Y (torch.Tensor): The transformed data in the principal component space.

        Returns:
            torch.Tensor: The reconstructed original data.

        Raises:
            AssertionError: If the PCA model has not been fit before calling this method.
        """
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
