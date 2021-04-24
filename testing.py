import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os

torch.random.manual_seed(42)
test = unittest.TestCase()

import hw1.transforms as hw1tf
import hw1.datasets as hw1datasets
import cs236781.plot as plot
import hw1.linear_classifier as hw1linear

#Old tests:
# image_shape = (3, 32, 64)
# num_classes = 3
# low, high = 0, 10
#
# # Generate some random images and check values
# X_ = None
# for i in range(100):
#     X, y = hw1datasets.random_labelled_image(image_shape, num_classes, low, high)
#     test.assertEqual(X.shape, image_shape)
#     test.assertIsInstance(y, int)
#     test.assertTrue(0 <= y < num_classes)
#     test.assertTrue(torch.all((X >= low) & (X < high)))
#     if X_ is not None:
#         test.assertFalse(torch.all(X == X_))
#     X_ = X
#
# plot.tensors_as_images([X, X_]);
#
# seeds = [42, 24]
# torch.random.manual_seed(seeds[0])
#
# # Before the context, the first seed affects the output
# data_pre_context = torch.randn(100, )
#
# with hw1datasets.torch_temporary_seed(seeds[1]):
#     # Within this context, the second seed is in effect
#     data_in_context = torch.randn(100, )
#
# # After the context, the random state should be restored
# data_post_context = torch.randn(100, )
# data_around_context = torch.cat([data_pre_context, data_post_context])
#
# # Use first seed, generate data in the same way but without changing context in the middle
# torch.random.manual_seed(seeds[0])
# data_no_context = torch.cat([torch.randn(100, ), torch.randn(100, )])
#
# # Identical results show that the context didn't affect external random state
# test.assertTrue(torch.allclose(data_no_context, data_around_context))
#
# # The data generated in the context should match what we would generate with the second seed
# torch.random.manual_seed(seeds[1])
# test.assertTrue(torch.allclose(data_in_context, torch.randn(100, )))
#
# # Test RandomImageDataset
#
# # Create the dataset
# num_samples = 500
# num_classes = 10
# image_size = (3, 32, 32)
# ds = hw1datasets.RandomImageDataset(num_samples, num_classes, *image_size)
#
# # You can load individual items from the dataset by indexing
# img0, cls0 = ds[139]
#
# # Plot first N images from the dataset with a helper function
# fig, axes = plot.dataset_first_n(ds, 9, show_classes=True, nrows=3)
#
# # The same image should be returned every time the same index is accessed
# for i in range(num_samples):
#     X, y = ds[i]
#     X_, y_ = ds[i]
#     test.assertEqual(X.shape, image_size)
#     test.assertIsInstance(y, int)
#     test.assertEqual(y, y_)
#     test.assertTrue(torch.all(X == X_))
#
# # Should raise if out of range
# for i in range(num_samples, num_samples + 10):
#     with test.assertRaises(ValueError):
#         ds[i]
#
# ds = hw1datasets.ImageStreamDataset(num_classes, *image_size)
#
# # This dataset can't be indexed
# with test.assertRaises(NotImplementedError):
#     ds[0]
#
# # There is no length
# with test.assertRaises(TypeError):
#     len(ds)
#
# # Arbitrarily stop somewhere
# stop = torch.randint(2 ** 11, 2 ** 16, (1,)).item()
#
# # We can iterate over it, indefinitely
# for i, (X, y) in enumerate(ds):
#     test.assertEqual(X.shape, image_size)
#     test.assertIsInstance(y, int)
#
#     if i > stop:
#         break
#
# print(f'Generated {i} images')
# test.assertGreater(i, stop)
#
#
# import itertools as it
# import hw1.knn_classifier as hw1knn
#
# def l2_dist_naive(x1, x2):
#     """
#     Naive distance calculation, just for testing.
#     Super slow, don't use!
#     """
#     dists = torch.empty(x1.shape[0], x2.shape[0], dtype=torch.float)
#     for i, j in it.product(range(x1.shape[0]), range(x2.shape[0])):
#         dists[i,j] = torch.sum((x1[i] - x2[j])**2).item()
#     return torch.sqrt(dists)
#
#
# # Test distance calculation
# x1 = torch.randn(12, 34)
# x2 = torch.randn(45, 34)
#
# dists = hw1knn.l2_dist(x1, x2)
# dists_naive = l2_dist_naive(x1, x2)
#
# test.assertTrue(torch.allclose(dists, dists_naive), msg="Wrong distances")


#region BiasTrickTest

# tf_btrick = hw1tf.BiasTrick()
#
# test_cases = [
#     torch.randn(64, 512),
#     torch.randn(2, 3, 4, 5, 6, 7),
#     torch.randint(low=0, high=10, size=(1, 12)),
#     torch.tensor([10, 11, 12])
# ]
#
# for x_test in test_cases:
#     xb = tf_btrick(x_test)
#     print('shape =', xb.shape)
#     test.assertEqual(x_test.dtype, xb.dtype, "Wrong dtype")
#     test.assertTrue(torch.all(xb[..., 1:] == x_test), "Original features destroyed")
#     test.assertTrue(torch.all(xb[..., [0]] == torch.ones(*xb.shape[:-1], 1)), "First feature is not equal to 1")
#endregion


#region linearClassifierCreator Test - Preperations
import torchvision.transforms as tvtf

# Define the transforms that should be applied to each image in the dataset before returning it
tf_ds = tvtf.Compose([
    # Convert PIL image to pytorch Tensor
    tvtf.ToTensor(),
    # Normalize each chanel with precomputed mean and std of the train set
    tvtf.Normalize(mean=(0.1307,), std=(0.3081,)),
    # Reshape to 1D Tensor
    hw1tf.TensorView(-1),
    # Apply the bias trick (add bias element to features)
    hw1tf.BiasTrick(),
])

import hw1.datasets as hw1datasets
import hw1.dataloaders as hw1dataloaders

# Define how much data to load
num_train = 10000
num_test = 1000
batch_size = 1000

# Training dataset
data_root = os.path.expanduser('~/.pytorch-datasets')
ds_train = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root=data_root, download=True, train=True, transform=tf_ds),
    num_train)

# Create training & validation sets
dl_train, dl_valid = hw1dataloaders.create_train_validation_loaders(
    ds_train, validation_ratio=0.2, batch_size=batch_size
)

# Test dataset & loader
ds_test = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root=data_root, download=True, train=False, transform=tf_ds),
    num_test)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size)

x0, y0 = ds_train[0]
n_features = torch.numel(x0)
n_classes = 10

# Make sure samples have bias term added
test.assertEqual(n_features, 28*28*1+1, "Incorrect sample dimension")

#endregion


# region Classifier tests

# # Create a classifier
# lin_cls = hw1linear.LinearClassifier(n_features, n_classes)
#
# # Evaluate accuracy on test set
# mean_acc = 0
# for (x,y) in dl_test:
#     y_pred, _ = lin_cls.predict(x)
#     mean_acc += lin_cls.evaluate_accuracy(y, y_pred)
# mean_acc /= len(dl_test)
#
# print(f"Accuracy: {mean_acc:.1f}%")

#endregion

import cs236781.dataloader_utils as dl_utils
from hw1.losses import SVMHingeLoss

torch.random.manual_seed(42)

# Classify all samples in the test set
# because it doesn't depend on randomness of train/valid split
x, y = dl_utils.flatten(dl_test)

# Compute predictions
lin_cls = hw1linear.LinearClassifier(n_features, n_classes)
y_pred, x_scores = lin_cls.predict(x)

# Calculate loss with our hinge-loss implementation
loss_fn = SVMHingeLoss(delta=1.)
loss = loss_fn(x, y, x_scores, y_pred)

# Compare to pre-computed expected value as a test
expected_loss = 9.0233
print("loss =", loss.item())
print('diff =', abs(loss.item()-expected_loss))
test.assertAlmostEqual(loss.item(), expected_loss, delta=1e-2)

# Create a hinge-loss function
loss_fn = SVMHingeLoss(delta=1)

# Compute loss and gradient
loss = loss_fn(x, y, x_scores, y_pred)
grad = loss_fn.grad()

# Sanity check only (not correctness): compare the shape of the gradient
test.assertEqual(grad.shape, lin_cls.weights.shape)