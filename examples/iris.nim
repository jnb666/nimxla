# logistic regression example using the Iris flowers dataset 

import std/[strutils, strformat, sequtils, logging, random, math, os, parsecsv]
import nimxla
import cligen

setPrintOpts(precision=4, minWidth=10, floatMode=ffDecimal, threshold=100, edgeItems=5)

type
  IrisData = object
    data:    Tensor[float32]
    labels:  Tensor[int64]
    classes: seq[string]

proc readIrisData(): IrisData =
  ## read CSV data and returns input data, labels and list of classes
  var p: CsvParser
  let file = joinPath(currentSourcePath.parentDir, "iris.data")
  p.open(file)
  p.readHeaderRow()
  var data: seq[float32]
  var labels: seq[int64]
  while p.readRow():
    var rec: seq[float32]
    for col in items(p.headers):
      let val = p.rowEntry(col)
      if col != "class":
        rec.add parseFloat(val)
      else:
        var label = result.classes.find(val)
        if label < 0:
          label = result.classes.len
          result.classes.add val      
        labels.add label.int64
    data.add rec
  p.close()
  result.data = data.toTensor().reshape(-1, p.headers.len-1)
  result.labels = labels.toTensor

proc initWeights(c: Client, nin, nout: int, seed: int64): Buffer =
  # initialise network weights
  if seed != 0:
    randomize(seed)
  else:
    randomize()
  let values = newSeqWith(nin*nout, gauss(sigma = 0.1).float32)
  return c.newBuffer(values.toTensor.reshape(nin, nout))

proc softmax*(a: Node, axis: int): Node =
  ## Softmax operation, shifted for numerical stability.
  let maxval = a.max([axis], keepDims=true)
  maxval.noGrad = true
  let exp_a = exp(a - maxval)
  let sum_a = exp_a.sum([axis], keepDims=true)
  result = exp_a / sum_a

proc crossEntropyLoss(b: Builder, pred, target: Node): Node =
  # calc loss from softmax output
  let shape = [target.dims[0], 1]
  let indices = concat(b.iota(I64, shape, axis=0), [target.reshape(shape)], axis=1)
  -sum(log(pred.gather(indices.reshape(-1, 1, 2))))

proc model(c: Client, batch, nin, nout: int): Executable =
  # build model to calc predictions, loss and weight gradients
  let b = newBuilder("iris")
  let input = b.parameter(F32, [batch, nin], "x")
  let weights = b.parameter(F32, [nin, nout], "w")
  let labels = b.parameter(I64, [batch], "y")
  let batchSize = b.constant(batch.float32)
  let output = softmax(dot(input, weights), 1)
  let loss = b.crossEntropyLoss(output, labels)
  debug "forward graph: ", loss.toString
  let grad = b.gradient(loss, ["w"])[0] / batchSize
  debug "weight grad: ", grad.toString
  let comp = b.build b.makeTuple(output, loss, grad)
  c.compile(comp)

proc sgd(c: Client, nin, nout: int, learnRate: float): Executable =
  let b = newBuilder("sgd")
  let weights = b.parameter(F32, [nin, nout], "weights")
  let grads = b.parameter(F32, [nin, nout], "grads")
  let comp = b.build(weights - b.constant(learnRate.float32) * grads)
  c.compile(comp)

proc calcAccuracy(c: Client, batch, nout: int): Executable =
  let b = newBuilder("accuracy")
  let predict = b.parameter(F32, [batch, nout], "predict")
  let labels = b.parameter(I64, [batch], "labels")
  let batchSize = b.constant(batch.float32)
  let comp = b.build sum(convert(predict.argmax(axis=1) == labels, F32)) / batchSize
  c.compile(comp)

proc printStats(epoch, batch: int, loss: float, predict, labels: Buffer, accFn: Executable) =
  let accuracy = accFn.run([predict, labels]).f32[] * 100
  info &"epoch {epoch:3}:  loss: {loss:8.4f}  accuracy: {accuracy:.1f}%"

proc train(epochs = 100, logEvery = 10, learnRate = 0.05, seed: int64 = 0, gpu = false, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = if gpu: newGPUClient() else: newCPUClient()
  echo c
  # get dataset
  let d = readIrisData()
  info "Iris dataset: " & $d.data.shape & "  classes: " & (d.classes)[2 .. ^2]
  let (batch, nin, nout) = (d.data.dims[0], d.data.dims[1], d.classes.len)
  let inputs = c.newBuffer(d.data)
  let labels = c.newBuffer(d.labels)

  # compile model executable and initialise weights
  let exec = c.model(batch, nin, nout) 
  var weights = c.initWeights(nin, nout, seed)
  info "initial weights: ", weights.f32
  # compile optimizer used to update the weights and function to calc the accuracy
  let optim = c.sgd(nin, nout, learnRate)
  let accFn = c.calcAccuracy(batch, nout)
  info "training with learning rate = ", learnRate

  for epoch in 1 .. epochs:
    # one step processing the data - both forward and backward pass
    var (pred, loss, grads) = exec.runAndUnpack([inputs, weights, labels]).tuple3
    debug "predict:", pred.f32
    let lossValue = loss.f32[] / batch.float
    if lossValue.isNan:
      error "loss returns Nan value - aborting"
      break
    # log stats to console
    if epoch mod logEvery == 0 or debug:
      printStats(epoch, batch, lossValue, pred, labels, accFn)
    # update weights 
    debug "grads:   ", grads.f32
    weights = optim.run([weights, grads])

  # run completed
  info "final weights: ", weights.f32


dispatch train

