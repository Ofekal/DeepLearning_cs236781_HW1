r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. False. In-sample error is the error rate we get on the data which used to build the predictor.
Since test set is kinda new data set, and since we don't use it to build the model, the errors made in 
test set are considered as out-of-sample error.

2. False. Though there is no optimal split percentage, not every split percentage is considered good.
The split percentage is expressed as a number between 0 and 1, for instance 0.7 means 70% of the data goes to train set, and 30% to test set.
If we take a small value, for example 0.1 for training and 0.9 for testing, the model will get low accuracy results since it has not been trained enough, 
and the diversity of the test set is too big relatively to the train set.
On the other hand, split percentage of 0.95 is also not good because it will cause a phenomenom known as over-fitting, meaning that the model is highly correlated with the train set.
The key is to find the balance between the two, based on experience, and considering the type of the data we use.

3. True. Cross-validation is a technique for evaluating models by training several models on subsets of the train data.
Basically, it's done to find the best model among the tested models.

4. True. Cross validation is a technique in which the train set is partitioned into k number of mini-sets. 
We train each of these validation-sets in order to find the hyperparameters that will lead to the best performance. 
We then assume that the model's performance will be similar to those tested in the CV proccess. 
Since the validation set was never used for training, and only for estimating our performance, it is considered as an unseen data. Therefore we can use the validation-set performance of each fold as a proxy for the model's generalization error.

"""

part1_q2 = r"""
**Your answer:**

Our friend's approach is not justified. Although adding regularization term can help to generalize the model and reduce over-fitting, the rest of his approach is basically wrong.
By testing $\lambda$'s performance on the test set, he just made sure that his model fits the specific test set he had. 
When checking on another test set, the hyperparameter he chose may not fit. His approach just made his model to be overfitted to the train **and** test sets he had, instead of just being over-fitted to the training set. 


"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

In general, increasing k does improve generalization as long as k is not over increased. The best choice of k depends upon the data and its size.
Generally, larger values of k reduce effect of the noise on the classification, for instance, for k=1 the results might be highly effected by outliers. On the other hand, high values of k make boundaries between classes less distinct, which means that we may include points from other classes into the neighborhood, for instance if we take it to the extreme, setting $k=len(data)$ will have no value for prediction.
In addition, large K means simple model, which is always considered as high biased with less variance (a small K model will have low bias and high variance). A too high value of K will cause a very smooth decision boundary (and even weak, loose, inaccurate) which will reduce accuracy eventually.

"""

part2_q2 = r"""
**Your answer:**

1. Using k-fold CV is better than selecting the best model with respect to train-set accuracy because doing so will lead to an over-fitted model.

2. To better explain our answer we will use an example: 
Let us imagine a predictor or dependent variable similar to this:

`2.3  2.4  2.5  2.6  2.7  2.4  18.0  15.0  2.6  3.0  2.7...`
 
Where everything seems fine with only two large outliers. If we have a small test set and just by chance the two outliers fall within that small test set they will influence that test set a lot and there is going to be a bad fit because both outliers contorted the test set.

If we split the same data into k sets, chances are higher that both outliers will not fall within the same group, and if they do, the bad situation with both outliers in the same small test set will affect only one out of k test runs. 
In other words, the test set uses fewer test cases than cross validation (which uses all available cases in turn) and is therefore subject to higher uncertainty.

```

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
Though $\lambda$ and $\Delta$ seems like different hyperparameters, they kind of have the same responsibility. $\Delta$ make sure that the correct class score is higher than the incorrect classes score by some fixed margin $\Delta$, and $\lambda$ is responsible of the magnitude of the weights. So $\Delta$ can be set to any positive number, but this number is basically meaningless, because after $\Delta$ is set, $\lambda$ can be set to any number and then shrink or stretch the differences arbitrarily. For example, if $\Delta$ was set to 1, and this margin existed in some sample, then the loss will not acculumate. But if some large $\lambda$ was set, then the margin got bigger for all weights including this specific sample, in a way that it's difference might be highly larger now than $\Delta$!
Therefore, the exact value of $\Delta > 0$ does not matter, and the more important hyperparameter is $\lambda$ which will determine how large we allow the weights to grow.

"""

part3_q2 = r"""
**Your answer:**

1. The image visualization of the weights can give us information about them. We can see that each element represent
a different digit. Generally speaking, the weights determines which features has more impact on the label it belongs to.
And viewing the image this can be understood - the weights learns which pixels are more important for each class.
For instance, we can see that the most left image has a round shape like 0 inside of it, and the one next to it is similar to the digit 1 etc.
Looking at the images, we can understand why the model made errors: several weights give more weight to the same pixels, meanind that they see similar pixels as important ones.

2. There are several similarities as well as differences:
The similarity comes when looking at the "closest similar sample". The KNN model search which other samples have similar features, and make the classification based on this information. 
SVM looks at the new sample and searches the closest weight that is similar to it. Thus the models are similar - when looking at a new sample, both search something similar.
Biggest difference for our opinion is the fact that the KNN model just memorizing the data, while SVM tries to make rules\learn a hyperplane that will make the classification. 


"""

part3_q3 = r"""
**Your answer:**

1. The learning rate we chose is good. The graphs shows a convergence which seems fair, not too steep nor too slow. 
A too high learning rate will show an "unsteady" behavior with sharp and steep steps. The steps were big and in order to converge, more epochs would be needed.
A too low learning rate on the other hand will show a very slow convergence (and there may not be one at all).


2. The accuracy graph of the train and validation set resembles that the model is slightly overfitted to the training set. The training set accuracy is slightly higher than the accuracy of the validation set, but not with a big difference.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern is a random pattern around the horizontal axis. such spreading suggests that the data is likely to have a good fit for linear regression.
In the top 5 features plot, we can see some random pattern around the horizontal axis but with quite extensive amount of outliers, which suggest that based only on the top 5 featuers the prediction will not be as accurate as we would wish it to be. In comparssion with the trained model (especially after adding non-linear featuers), we can observe decreasing amount of outliers and better spreading around the horizontal axis that implies a better fit for linear regression.
"""

part4_q2 = r"""
**Your answer:**

1. Yes. this is still a linear regression model. 
   When we use a nonlinear feature we "know" that it isn't linear, however for the model this fact is "unknown", and the variable is simply another variable (with some correlation to the linear variable that it is taken from). in more mathematically way-
   if we take for instance quadratic as a function of x linear in the coefficients a, b and c. 
   $y=a+bx+cx^2$
   the equation isn't linear regarding to x but is linear in the coefficients a, b and c.
   More generally, a general linear model can be expressed as  $y=∑a_ih_i(x)$, where the $h_i$ are arbitrary functions of vectorial inputs x
2. Yes, if there is a nonlinear connection, we can fit a nonlinear function for the data. however, we should be aware of possible increasing in the overfitting of the model.
3. The decision boundary will be a bit more flexible and hopefully more accurate. As we explained in Q2, the model still behaves as linear regression and therefore can be presented as a hyperplane.

"""

part4_q3 = r"""
**Your answer:**

1. Numpy linspace returns evenly spaced numbers over a linear scale. Numpy logspace returns numbers spaced evenly over a log scale.
We are looking to find the value of $\lambda$ that best fits for us. Using logspace allow us to explore the range of values in the scale we are interested in. In our case, we are checking what value of $\lambda$ is best and we are fine-tuning it. Using linspace instead of logspace would have force us to sample a very big amount of values.

2. The number of times in total the was model fitted to data is given by:
$ k folds * lambda range * degree range $

while $k folds = 3;     lambda range = 20;     degree range = 3; $

So in total:  180 times.
"""

# ==============
