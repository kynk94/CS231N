import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  for i in range(X.shape[0]):
    score = np.dot(X[i],W).reshape(1,-1)
    exp_score = np.exp(score - np.max(score, keepdims=True))
    probs = exp_score/np.sum(exp_score, keepdims=True)
    loss += -np.log(probs[0,y[i]])
    dscores = probs.copy()
    dscores[0,y[i]] -= 1
    dW += X[i].reshape(1,-1).T.dot(dscores)
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # 0 <= probs <= 1
  correct_log_probs = -np.log(probs[np.arange(X.shape[0]), y]) # -log(<1) > 0
  loss = np.sum(correct_log_probs) / X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  dscores = probs.copy()
  dscores[np.arange(X.shape[0]), y] -= 1
  dscores /= X.shape[0]
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
