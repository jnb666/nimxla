## The nn module provides higher level functions for building neural network models using the nimxla graph API.

import std/[math, random, strutils, sequtils, sugar, tables, logging]
import ../nimxla

type
  Variable* = object
    ## A Variable is a learnable parameter associated with a module.
    name*: string
    data*: Buffer

  InitFunc* = proc(c: Client, dims: openarray[int], dtype: DataType): Buffer
    ## InitFunc defines a function used to initialise a learnable parameter.

  Module* = object
    ## A module holds the state for a set of variables together with a forward 
    ## computation which depends on an input value.
    builder*:   Builder
    variables*: seq[Variable]
    forward*:   proc(x: Node): Node


proc constantInit*(value: SomeFloat): InitFunc =
  ## Create a new buffer with a constant value.
  proc(c: Client, dims: openarray[int], dtype: DataType): Buffer =
    let t = fill(dims, value)
    c.newBuffer(t.toLiteral.convert(dtype))

proc uniformInit*(min = 0.0, max = 1.0): InitFunc =
  ## Create a new buffer with uniform random values from min to max.
  proc(c: Client, dims: openarray[int], dtype: DataType): Buffer =
    let t = newSeqWith(prod(dims), rand(min..max)).toTensor.reshape(dims)
    c.newBuffer(t.toLiteral.convert(dtype))

proc normalInit*(mean = 0.0, stddev = 1.0): InitFunc =
  ## Create a new buffer with random values chosen from a normal distrubution.
  proc(c: Client, dims: openarray[int], dtype: DataType): Buffer =
    let t = newSeqWith(prod(dims), gauss(mean, stddev)).toTensor.reshape(dims)
    c.newBuffer(t.toLiteral.convert(dtype))

proc newVariable*(c: Client, name: string, dims: openarray[int], dtype: DataType, init: InitFunc): Variable =
  ## Allocate a new variable with the given name and shape and initialise values.
  Variable(name: name, data: c.init(dims, dtype))

proc param*(b: Builder, p: Variable, suffix = ""): Node =
  ## Create a new parameter node from this Variable
  let s = p.data.shape
  b.parameter(s.dtype, s.dims, p.name & suffix)

proc add*(m: var Module, modules: varargs[Module]) =
  ## Add variables from modules to m.
  for sub in modules:
    m.variables.add sub.variables

proc varNames*(m: Module): seq[string] =
  ## List of all the variable names defined for this module.
  map(m.variables, x => x.name)

proc gradNames*(m: Module): seq[string] =
  ## List of all the variable gradient output names for this module.
  map(m.variables, x => x.name & "_grad")

proc setParams*(m: Module, params: var Params) =
  ## Add model variables to the params table
  for p in m.variables:
    params[p.name] = p.data

proc mseLoss*(pred, target: Node): Node =
  ## Mean square error loss function.
  let n = pred.builder.constant(math.prod(pred.dims), pred.dtype)
  sum((pred - target) * (pred - target)) / n

proc crossEntropyLoss*(pred, target: Node): Node =
  ## Cross entropy loss function calculated from softmax output.
  ## Pred should be predicted values with shape [n, classes] while target is a
  ## 1d vector of I64 labels each in range 0..classes.
  let shape = [target.dims[0], 1]
  let indices = concat(pred.builder.iota(I64, shape, axis=0), [target.reshape(shape)], axis=1)
  -sum(log(pred.gather(indices.reshape(-1, 1, 2))))

proc softmax*(a: Node, axis: int): Node =
  ## Softmax operation, shifted for numerical stability.
  let maxval = a.max([axis], keepDims=true)
  maxval.noGrad = true
  let exp_a = exp(a - maxval)
  let sum_a = exp_a.sum([axis], keepDims=true)
  result = exp_a / sum_a

proc initLinear*(c: Client, b: Builder, id: string, nin, nout: int, 
                 weights: InitFunc, biases = constantInit(0f32), dtype = F32): Module =
  ## Create a new fully connected linear layer with the given unique id and number of inputs and outputs.
  ## Weight parameters are initialised using the weights function. If biases is not nil then bias parameters 
  ## are initialised using this function and added to the output.
  result.builder = b
  let weight = c.newVariable(id & ".w", [nin, nout], dtype, weights)
  let W = b.param(weight)
  result.variables.add weight
  if biases == nil:
    result.forward = proc(x: Node): Node = dot(x, W)
  else:
    let bias = c.newVariable(id & ".b", [1, nout], dtype, biases)
    let B = b.param(bias)
    result.variables.add bias
    result.forward = proc(x: Node): Node = dot(x, W) + B

proc buildAndCompile*(c: Client, m: Module, input: Node, lossFn: proc(y: Node): Node): Executable =
  ## Build the execution graph for the given module and compile it to an executable.
  ## The executable returns a tuple with the following outputs:
  ## - pred: result of `m.forward(input)`
  ## - loss: result of `lossFn(pred)`
  ## - <v1.name>_grad, ...: gradients for each input variable with respect to the loss
  let b = m.builder
  let pred = m.forward(input)
  debug "forward function: ", pred.toString
  let loss = lossFn(pred)
  let grads = b.gradient(loss, m.varNames)
  let comp = b.build b.makeTuple(@[pred, loss] & grads)
  c.compile(comp, @["pred", "loss"] & m.gradNames)

proc buildSGD(m: Module, learnRate, weightDecay, momentum: float): (Computation, seq[string]) = 
  let b = newBuilder("sgd")
  var outputs: seq[Node]
  var names: seq[string]
  for p in m.variables:
    let x = b.param(p)
    var dx = b.param(p, "_grad")
    if weightDecay != 0 and not p.name.endsWith(".b"):
      dx = dx + b.constant(weightDecay, x.dtype) * x
    if momentum != 0:
      let prev = b.param(p, "_mom")
      let mu = b.constant(momentum, x.dtype)
      dx = mu * prev + dx
      names.add p.name & "_mom"
      outputs.add dx
    let lr = b.constant(learnRate, x.dtype)
    names.add p.name
    outputs.add x - lr*dx
  (b.build b.makeTuple(outputs), names)

proc optimSGD*(c: Client, m: Module, learnRate: float, weightDecay = 0.0, momentum = 0.0): proc(params: Params): seq[Variable] =
  ## Builds and compiles a stochastic gradient descent optimizer to optimize the variables for module m.
  ## This returns a function which takes as input a table with all of the named weight and gradient parameters and
  ## returns the list of model weight variables.
  let (comp, names) = buildSGD(m, learnRate, weightDecay, momentum)
  var state = initParams([])
  if momentum != 0:
    for p in m.variables:
      let s = p.data.shape
      state[p.name & "_mom"] = c.newBuffer(s.dtype, s.dims)
  let exec = c.compile(comp, names)

  proc (params: Params): seq[Variable] =
    var vars = params
    for key, val in state:
      vars[key] = val
    exec.runWith(vars)
    for key in state.keys:
      state[key] = vars[key]
    var res: seq[Variable]
    for i, name in m.varNames:
      res.add Variable(name: name, data: vars[name])
    return res


