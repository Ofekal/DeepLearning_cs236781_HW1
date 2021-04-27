import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # Create a tensor that stores just the score of the correct class of each sample:
        ground_truth_classes_scores = x_scores.gather(dim = 1, index = y.reshape(y.shape[0],1))
        # m_ij_matrix is the expression inside Li(W)'s Sigma.
        m_ij = self.delta + x_scores - ground_truth_classes_scores
        # zero matrix in the size of m_ij_matrix
        zero = torch.zeros_like(m_ij)
        # Li_w is the the same as the hinge loss formula.
        Li_w = torch.where(m_ij>0, m_ij,zero)
        # The final loss function, L(W) is:
        loss = Li_w.sum()/y.shape[0] - self.delta
        # ========================
        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['M'] = m_ij
        self.grad_ctx['X'] = x
        self.grad_ctx['Y'] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # prepare parameters:
        x = self.grad_ctx['X']
        m_ij = self.grad_ctx['M']
        y = self.grad_ctx['Y']
        zero_t = torch.zeros_like(m_ij)
        one_t = torch.ones_like(m_ij)
        # a tensor that describe the indicator function. store 1 if m_ij is positive, else 0.
        indicator = torch.where(m_ij > 0, one_t, zero_t)
        # sums up the occurrences of 1 (ones) in indicator.
        sigma = torch.sum(indicator, dim=1)
        # subtract sigma from the indexes in which the labels are correct:
        indicator[range(indicator.shape[0]), y] -= sigma
        # calculate X^T * G like the hint said, and then divide it by N, number of samples.
        grad = torch.mm(x.t(), indicator) / x.shape[0]

        # ========================

        return grad
