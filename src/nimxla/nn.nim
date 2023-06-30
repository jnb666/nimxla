## The nn module provides higher level functions for building neural network models using the nimxla graph API.

import std/[math, random, times, strutils, strformat, sequtils, sugar, tables, logging]
import ../nimxla

type
  Variable* = object
    ## A Variable is a learnable parameter associated with a module.
    name*: string
    data*: Buffer

  InitFunc* = proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer
    ## InitFunc defines a function used to initialise a learnable parameter.

  Optimizer* = proc(params: Params): seq[Variable]
    ## Optimizer function to update the model weights.

  Module* = object
    ## A module holds the state for a set of variables together with a forward 
    ## computation which depends on an input value. The training flag is set
    ## to true when in training mode.
    variables*: seq[Variable]
    forward*: proc(x: Node, training = false): Node
    info*: string


proc initRandom*(seed: int64 = 0): Rand =
  ## Initialize random generator. Picks a random seed from current time if seed == 0.
  let seed = if seed == 0:
    getTime().toUnix * 1_000_000_000 + getTime().nanosecond
  else:
    seed
  echo &"initRandom: seed={seed}"
  result = initRand(seed)

proc constantInit*(value: SomeFloat): InitFunc =
  ## Create a new buffer with a constant value.
  proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer =
    let t = fill(dims, value)
    c.newBuffer(t.toLiteral.convert(dtype))

proc uniformInit*(min = 0.0, max = 1.0): InitFunc =
  ## Create a new buffer with uniform random values from min to max.
  proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer =
    let t = newSeqWith(prod(dims), rng.rand(min..max)).toTensor.reshape(dims)
    c.newBuffer(t.toLiteral.convert(dtype))

proc normalInit*(mean = 0.0, stddev = 1.0): InitFunc =
  ## Create a new buffer with random values chosen from a normal distrubution.
  proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer =
    let t = newSeqWith(prod(dims), rng.gauss(mean, stddev)).toTensor.reshape(dims)
    c.newBuffer(t.toLiteral.convert(dtype))

proc getFanInOut(dims: openarray[int]): (int, int) =
  ## Get effective number of inputs and outputs.
  assert dims.len >= 2
  if dims.len == 2:
    # linear layer
    (dims[0], dims[1])
  else:
    # conv layer with channels last layout
    let ksize = prod(dims[1..^2])
    (dims[^1]*ksize, dims[0]*ksize)

proc glorotInit*(gain = 1.0): InitFunc =
  ## Initialisation function with values set in uniform range from
  ## ```
  ##  -gain*sqrt(6/(nin+nout)) to gain*sqrt(6/(nin+nout))
  ## ```
  ## where nin are effective number of inputs and outputs.
  ## Also known as Xavier uniform init.
  proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer =
    let (nin, nout) = getFanInOut(dims)
    let max = gain * sqrt(6.0 / (nin+nout).float)
    uniformInit(-max, max)(c, dims, dtype, rng)

proc heInit*(mean = 0.0, gain = 2.0, fanOut = false): InitFunc =
  ## Initialisation function with values from normal distrubution with std deviation as
  ## ```
  ##   sqrt(gain/nin) or sqrt(gain/nout) if fanOut is set
  ## ```
  ## where nin are effective number of inputs and outputs. Gain of 2 is suitable for Relu non-linearity.
  ## Also known as Kaiming normal init.
  proc(c: Client, dims: openarray[int], dtype: DataType, rng: var Rand): Buffer =
    let (nin, nout) = getFanInOut(dims)
    let stddev = if fanOut:
      sqrt(gain / nout.float)
    else:
      sqrt(gain / nin.float)
    normalInit(mean, stddev)(c, dims, dtype, rng)

proc newVariable*(c: Client, name: string, dims: openarray[int], dtype: DataType, init: InitFunc, rng: var Rand): Variable =
  ## Allocate a new variable with the given name and shape and initialise values.
  Variable(name: name, data: c.init(dims, dtype, rng))

proc param*(b: Builder, p: Variable, suffix = ""): Node =
  ## Create a new parameter node from this Variable
  let s = p.data.shape
  b.parameter(s.dtype, s.dims, p.name & suffix)

proc add*(m: var Module, modules: varargs[Module]) =
  ## Add variables from modules to m.
  var info: seq[string]
  for sub in modules:
    m.variables.add sub.variables
    info.add sub.info
  m.info = if m.info == "":
    info.join("\n")
  else:
    m.info & "\n" & indent(info.join("\n"), 2)

proc `$`*(m: Module): string =
  m.info

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
  ## Mean cross entropy loss function calculated from softmax output.
  ## Pred should be predicted values with shape [n, classes] while target is a
  ## 1d vector of I64 labels each in range 0..classes.
  let shape = [target.dims[0], 1]
  let indices = concat(pred.builder.iota(target.dtype, shape, axis=0), [target.reshape(shape)], axis=1)
  let nitems = pred.builder.constant(target.dims[0], pred.dtype)
  -sum(log(pred.gather(indices.reshape(-1, 1, 2)))) / nitems

proc softmax*(a: Node, axis = -1): Node =
  ## Softmax operation, shifted for numerical stability.
  let maxval = a.max([axis], keepDims=true)
  maxval.noGrad = true
  let exp_a = exp(a - maxval)
  let sum_a = exp_a.sum([axis], keepDims=true)
  result = exp_a / sum_a

proc dropout*(a: Node, ratio: float, training: bool, normalize = true): Node =
  ## Dropout randomly replaces ratio fraction of input values with zeros when training is true
  ## else if a no-op in test mode. If normalize is set then the output is scaled by `1 / (1-ratio)`
  if not training or ratio <= 0:
    return a
  let b = a.builder
  let ty = a.dtype
  let r = b.constant(ratio, ty)
  let rnd = rngUniform(b.zero(ty), b.one(ty), a.dims)
  result = select(rnd < r, b.zero(ty, a.dims),  a)
  if normalize:
    result = result / (b.one(ty) - r)

proc initLinear*(c: Client, rng: var Rand, id: string, nin, nout: int,
                 weights = heInit(), biases = constantInit(0.0), dtype = F32): Module =
  ## Create a new fully connected linear layer with the given unique id and number of inputs and outputs.
  ## Weight parameters are initialised using the weights function. If biases is not nil then bias parameters 
  ## are initialised using this function and added to the output.
  result.info = &"{id}: linear(nin={nin}, nout={nout}, bias={biases != nil})"
  let weight = c.newVariable(id & ".w", [nin, nout], dtype, weights, rng)
  result.variables.add weight
  if biases == nil:
    result.forward = proc(x: Node, training = false): Node =
      let W = x.builder.param(weight)
      dot(x, W)
  else:
    let bias = c.newVariable(id & ".b", [1, nout], dtype, biases, rng)
    result.variables.add bias
    result.forward = proc(x: Node, training = false): Node =
      let W = x.builder.param(weight)
      let B = x.builder.param(bias)
      dot(x, W) + B

proc initConv2d*(c: Client, rng: var Rand, id: string, inChannels, outChannels: int, kernelSize: Opt2d, 
                 strides: Opt2d = 1, padding: Pad2d = pad(0), dilation: Opt2d = 1, groups = 1, 
                 weights = heInit(), biases = constantInit(0.0), dtype = F32): Module =
  ## Create a new 2 dimensional convolution layer with parameters in channels last format.
  ## Weight parameters are initialised using the weights function. If biases is not nil then bias parameters 
  ## are initialised using this function and added to the output.
  result.info = &"{id}: conv2d(inChannels={inChannels}, outChannels={outChannels}, kernelSize={kernelSize}"
  if strides != 1: result.info.add &", strides={strides}"
  if padding != pad(0): result.info.add &", padding={padding}"
  if dilation != 1: result.info.add &", dilation={dilation}"
  if groups != 1: result.info.add &", groups={groups}"
  result.info.add &", bias={biases != nil})"

  let wdims = @[outChannels] & kernelSize.seq2 & @[inChannels]
  let weight = c.newVariable(id & ".w", wdims, dtype, weights, rng)
  result.variables.add weight
  if biases == nil:
    result.forward = proc(x: Node, training = false): Node =
      let W = x.builder.param(weight)
      conv2d(x, W, strides, padding, dilation, groups)
  else:
    let bias = c.newVariable(id & ".b", @[1, 1, 1, outChannels], dtype, biases, rng)
    result.variables.add bias
    result.forward = proc(x: Node, training: bool): Node =
      let W = x.builder.param(weight)
      let B = x.builder.param(bias)
      conv2d(x, W, strides, padding, dilation, groups) + B

proc compileTest*(c: Client, m: Module, input: Node): Executable =
  ## Build the execution graph for the given module and compile it to an executable.
  ## The executable returns the output from `m.forward(input)`
  let pred = m.forward(input, training=false)
  let comp = input.builder.build(pred)
  c.compile(comp, ["pred"])

proc compileTrain*(c: Client, m: Module, input: Node, lossFn: proc(y: Node): Node): Executable =
  ## Build the execution graph for the given module and compile it to an executable.
  ## The executable returns a tuple with the following outputs:
  ## - pred: result of `m.forward(input)`
  ## - loss: result of `lossFn(pred)`
  ## - <v1.name>_grad, ...: gradients for each input variable with respect to the loss
  let b = input.builder
  let pred = m.forward(input, training=true)
  debug "forward function: ", pred.toString
  let loss = lossFn(pred)
  let grads = b.gradient(loss, m.varNames)
  let comp = b.build b.makeTuple(@[pred, loss] & grads)
  c.compile(comp, @["pred", "loss"] & m.gradNames)

proc makeOptimizer(m: Module, exec: Executable, init: proc():Params): Optimizer =
  var state = init()
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

proc buildSGD(c: Client, m: Module, learnRate, weightDecay, momentum: float, nesterov: bool): Executable =
  let b = newBuilder("sgd")
  var outputs: seq[Node]
  var names: seq[string]
  let lr = b.constant(learnRate, F32)
  let decay = b.constant(weightDecay, F32)
  let mu = b.constant(momentum, F32)
  for p in m.variables:
    let x = b.param(p).convert(F32)
    var dx = b.param(p, "_grad").convert(F32)
    if weightDecay != 0 and not p.name.endsWith(".b"):
      dx = dx + decay * x
    if momentum != 0:
      let prev = b.param(p, "_mom")
      let bt = mu * prev + dx
      names.add p.name & "_mom"
      outputs.add bt
      if nesterov:
        dx = dx + mu * bt
      else:
        dx = bt
    names.add p.name
    outputs.add x - lr*dx
  let comp = b.build b.makeTuple(outputs)
  c.compile(comp, names)

proc optimSGD*(c: Client, m: Module, learnRate: float, weightDecay = 0.0, 
               momentum = 0.0, nesterov = false): Optimizer =
  ## Builds and compiles a stochastic gradient descent optimizer to optimize the variables for module m.
  ## This returns a function which takes as input a table with all of the named weight and gradient parameters and
  ## returns the list of model weight variables.
  let exec = c.buildSGD(m, learnRate, weightDecay, momentum, nesterov)
  let init = proc(): Params =
    var state = initParams([])
    if momentum != 0:
      for p in m.variables:
        let s = p.data.shape
        state[p.name & "_mom"] = c.newBuffer(s.dtype, s.dims)
    return state
  makeOptimizer(m, exec, init)

proc buildAdam(c: Client, m: Module, learnRate, weightDecay, beta1, beta2, epsilon: float): Executable = 
  let b = newBuilder("adam")
  var outputs: seq[Node]
  var names: seq[string]
  let lr = b.constant(learnRate, F32)
  let eps = b.constant(epsilon, F32)
  let decay = b.constant(weightDecay, F32)
  let b1 = b.constant(beta1, F32)
  let b1t = b.parameter(F32, name = "beta1_t")
  names.add "beta1_t"
  outputs.add b1t * b1
  let b2 = b.constant(beta2, F32)
  let b2t = b.parameter(F32, name = "beta2_t")
  names.add "beta2_t"
  outputs.add b2t * b2
  let one = b.one(F32)
  for p in m.variables:
    let x = b.param(p).convert(F32)
    var dx = b.param(p, "_grad").convert(F32)
    if weightDecay != 0 and not p.name.endsWith(".b"):
      dx = dx + decay * x
    let mtPrev = b.param(p, "_mt")
    let mt = b1 * mtPrev + (one-b1) * dx
    let mtp = mt / (one - b1t)
    names.add p.name & "_mt"
    outputs.add mt
    let vtPrev = b.param(p, "_vt")
    let vt = b2 * vtPrev + (one-b2) * dx*dx
    let vtp = vt / (one - b2t)
    names.add p.name & "_vt"
    outputs.add vt
    names.add p.name
    outputs.add x - lr*mtp/(sqrt(vtp) + eps)
  let comp = b.build b.makeTuple(outputs)
  c.compile(comp, names)

proc optimAdam*(c: Client, m: Module, learnRate: float, weightDecay = 0.0, 
                beta1 = 0.9, beta2 = 0.999, eps = 1e-8): Optimizer =
  ## Builds and compiles an optimizer using the Adam algorithm.
  ## This returns a function which takes as input a table with all of the named weight and gradient parameters and
  ## returns the list of model weight variables.
  let exec = c.buildAdam(m, learnRate, weightDecay, beta1, beta2, eps)
  let init = proc(): Params =
    var state = initParams({
      "beta1_t": c.newBuffer(lit(beta1.float32)),
      "beta2_t": c.newBuffer(lit(beta2.float32))
    })
    for p in m.variables:
      let s = p.data.shape
      state[p.name & "_mt"] = c.newBuffer(s.dtype, s.dims)
      state[p.name & "_vt"] = c.newBuffer(s.dtype, s.dims)
    return state
  makeOptimizer(m, exec, init)
