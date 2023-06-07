## The nn module provides higher level functions for building neural network models using the nimxla graph API.

import std/math
import ../nimxla

type
  Parameter* = object
    name: string
    data: Buffer

  Module* = object
    forward: proc(x: Node): Node
    params:  seq[Parameter]


proc initLinear*(b: Builder, id: string, nin, nout: int, dtype=F32): Module =
  ## Create a new fully connected linear layer with the given unique id 
  ## and number of inputs and outputs.
  let weight = b.parameter(dtype, [nin, nout], "w"&id)
  let bias  = b.parameter(dtype, [1, nout], "b"&id)
  result.forward = proc(x: Node): Node = dot(x, weight) + bias

proc softmax*(a: Node, axis: int): Node =
  ## Softmax operation, shifted for numerical stability.
  let maxval = a.max([axis], keepDims=true)
  maxval.noGrad = true
  let exp_a = exp(a - maxval)
  let sum_a = exp_a.sum([axis], keepDims=true)
  result = exp_a / sum_a

proc crossEntropyLoss*(b: Builder, pred, target: Node): Node =
  ## Cross entropy loss function calculated from softmax output.
  ## Pred should be predicted values with shape [n, classes] while target is a
  ## 1d vector of I64 labels each in range 0..classes.
  let shape = [target.dims[0], 1]
  let indices = concat(b.iota(I64, shape, axis=0), [target.reshape(shape)], axis=1)
  -sum(log(pred.gather(indices.reshape(-1, 1, 2))))

proc mseLoss*(b: Builder, pred, target: Node): Node =
  ## Mean square error loss function.
  let n = b.constant(math.prod(pred.dims), pred.dtype)
  sum((pred - target) * (pred - target)) / n
