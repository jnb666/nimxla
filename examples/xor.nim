# 2 layer MLP test using xor model

import std/[strutils, strformat, sugar, logging, random, math, tables]
import nimxla
import nimxla/nn
import cligen

let data = @@[[0f32, 0], [0, 1], [1, 0], [1, 1]]
let target = @@[[0f32], [1], [1], [0]]

setPrintOpts(precision=4, minWidth=10, floatMode=ffDecimal)

proc initModel(c: Client): (Module, Executable) =
  let b = newBuilder("xor")
  let layer1 = c.initLinear("1", nin=2, nout=2, weights = uniformInit())
  let layer2 = c.initLinear("2", nin=2, nout=1, weights = uniformInit())
  var model: Module
  model.add(layer1, layer2)
  model.forward = proc(x: Node): Node =
    let l1 = layer1.forward(x).sigmoid
    layer2.forward(l1)
  let x = b.parameter(F32, data.dims, "x")
  let y = b.parameter(F32, target.dims, "y")
  let exec = c.compileTrain(model, x, yp => mseLoss(yp, y))
  return (model, exec)

proc calcAccuracy(c: Client): Executable =
  let b = newBuilder("accuracy")
  let y = b.parameter(F32, target.dims, "y")
  let val = select(y >= b^0.5f32, b^1f32, b^0f32)
  let correct = convert(val == b.constant(target), F32).sum
  let comp = b.build correct / b.constant(target.len, F32)
  c.compile(comp)

proc printStats(epoch: int, loss: float, predict: Buffer, accFn: Executable) =
  let accuracy = accFn.run([predict]).f32[] * 100
  echo &"epoch {epoch:4}:  loss: {loss:8.4f}  accuracy: {accuracy:.1f}%"

proc train(epochs = 1000, logEvery = 10, learnRate = 0.2, minLoss = 0.1, seed: int64 = 0, gpu = false, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = if gpu: newGPUClient() else: newCPUClient()
  echo c
  # compile model executable and initialise parameters
  if seed != 0:
    randomize(seed)
  else:
    randomize()
  var params = initParams({"x": c.newBuffer(data), "y": c.newBuffer(target)})
  var (model, exec) = c.initModel()
  for p in model.variables:
    echo &"initial {p.name}: {p.data.f32}"
  # compile optimizer used to update the weights and function to calc the accuracy
  let optim = c.optimSGD(model, learnRate, momentum = 0.9)
  let accFn = c.calcAccuracy()

  echo "training with learning rate = ", learnRate
  for epoch in 1 .. epochs:
    # one step processing the data - both forward and backward pass
    model.setParams(params)
    exec.runWith(params)
    debug params["pred"]
    let loss = params["loss"].f32[]
    if loss.isNan:
      error "loss returns Nan value - aborting"
      break
    # log stats to console
    if loss <= minLoss or epoch mod logEvery == 0 or debug:
      printStats(epoch, loss, params["pred"], accFn)
      if loss <= minLoss:
        break
    # update weights
    model.variables = optim(params)
 
  for p in model.variables:
    echo &"final {p.name}: {p.data.f32}"


dispatch train

