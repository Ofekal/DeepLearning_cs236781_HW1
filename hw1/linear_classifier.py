import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        standard_mean = 0.0
        tensor_to_fill = torch.empty(n_features, n_classes)
        self.weights = tensor_to_fill.normal_(mean = standard_mean,std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        multiply_tensor = torch.mm(x, self.weights)
        # dims = multiply_tensor.size()
        y_pred = torch.argmax(multiply_tensor, 1)
        class_scores = multiply_tensor
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        correct = torch.eq(y,y_pred).tolist()
        acc = (correct.count(True))/len(correct)
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            number_of_batches = 0
            total_loss = 0
            for x_train, y_train in dl_train:
                # get the scores of prediction and the y_pred for train group:
                y_pred_train, class_scores_train = self.predict(x_train)
                # sum total loss (loss for all batches)
                total_loss += loss_fn.loss(x_train, y_train, class_scores_train, y_pred_train)
                # update the weights according to weight_decay and learning rate
                self.weights = self.weights - learn_rate * loss_fn.grad() - weight_decay * self.weights
                # sum total accuracy (accuracy for all batches)
                total_correct += self.evaluate_accuracy(y_train, y_pred_train)
                # count number of batches:
                number_of_batches += 1
            # get the averages and add them to the lists:
            average_loss = total_loss / number_of_batches
            train_res.accuracy.append(average_loss)
            train_res.loss.append(total_loss/number_of_batches)

            # Repeat scheme with validation set:
            total_loss = 0
            number_of_batches = 0
            average_loss = 0

            for x_val, y_val in dl_valid:
                y_pred_val, class_scores_val = self.predict(x_val)
                self.weights = self.weights - learn_rate * loss_fn.grad() - weight_decay * self.weights
                total_correct += self.evaluate_accuracy(y_val, y_pred_val)
                total_loss += loss_fn.loss(x_val, y_val, class_scores_val, y_pred_val)
                number_of_batches += 1
            average_loss = total_loss / number_of_batches
            train_res.accuracy.append(average_loss)
            train_res.loss.append(total_loss / number_of_batches)

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if not has_bias:
            w_temporary = self.weights
        else:
            w_temporary = self.weights[1:]
        w_images = w_temporary.t().reshape(self.n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.05
    hp['learn_rate'] = 0.1
    hp['weight_decay'] = 0.05
    # ========================

    return hp
