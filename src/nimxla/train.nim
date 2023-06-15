## The train module has common functions for training neural networks.
## It depends on the nn and data modules in addition to the core nimxla modules.

import std/[tables, json, logging, math, strformat, sugar, monotimes]
import ../nimxla
import nn, data, plots
import ws

type
  Trainer* = object
    ## Trainer object holds the state for a training
    client*:   Client
    optim*:    Optimizer
    trainer*:  Executable
    tester*:   Executable
    trainAcc*: Executable
    testAcc*:  Executable
    stats*:    Table[string, seq[float]]
    predict*:  Tensor[int32]

let
  plotNames = @[@["loss"], @["train", "test"]]
  plotLayout = gridLayout(2, 1, ["loss", "accuracy"], ["epoch"])


proc trainFunc*(c: Client, model: Module, dtype: DataType, shape: seq[int], lossFn: proc(yp, y: Node): Node): Executable =
  ## Compile training function with given input data shape and loss function which is applied to the output.
  debug &"trainFunc: shape={shape} dtype={dtype}"
  let b = newBuilder("trainer")
  let x = b.parameter(dtype, shape, "x")
  let y = b.parameter(I32, [shape[0]], "y")
  c.compileTrain(model, x, yp => lossFn(yp, y))

proc testFunc*(c: Client, model: Module, dtype: DataType, shape: seq[int]): Executable =
  ## Compile test function with given input data shape.
  debug &"testFunc: shape={shape}"
  let b = newBuilder("tester")
  let x = b.parameter(dtype, shape, "x")
  c.compileTest(model, x)

proc accuracyFunc*(c: Client, batch, nout: int, outType=F32, labelType=I32): Executable =
  ## Helper function to calculate the accuracy from a set of predictions.
  ## Callable has two input parameters
  ## - model output array of shape <outType>[batch, nout]
  ## - target labels of shape <labelType>[batch]
  ## And tuple with two outputs
  ## - labels array of predicted class for each sample, and
  ## - accuracy F32 scalar in range 0-1 from comparison with labels
  debug &"calcAccuracy batch={batch} nout={nout}"
  let b = newBuilder("accuracy")
  let predict = b.parameter(outType, [batch, nout], "pred")
  let targets = b.parameter(labelType, [batch], "y")
  let labels = predict.argmax(axis=1, ixType=labelType)
  let accuracy = mean(convert(labels == targets, F32))
  let comp = b.build(b.makeTuple(labels, accuracy))
  c.compile(comp, @["labels", "accuracy"])

proc trainEpoch*[T: ElemType](t: var Trainer, model: var Module, loader: DataLoader): (float, float) =
  ## Train on one epoch of batches of data from the training set, returns average loss and accuracy on training dara
  ## T should be the type of data returned from the loader. Output loss should be a float32 scalar.
  var data = newTensor[T](loader.shape)
  var targets = newTensor[int32](loader.batchSize)
  var avgLoss = 0.0
  var accuracy = 0.0
  for batch in getBatch(loader, data, targets):
    var params = initParams({"x": t.client.newBuffer(data), "y": t.client.newBuffer(targets)})
    model.setParams(params)
    t.trainer.runWith(params)
    t.trainAcc.runWith(params)
    let loss = params["loss"].f32[].float
    let accVal = params["accuracy"].f32[].float
    if batch mod 10 == 0:
      debug &"train batch {batch}: loss = {loss:.2f}  accuracy = {accVal:.3f}"
      debug "  tgt: ", targets
      debug "  pred:", params["labels"].i32
    if loss.isNan:
      return (loss, 0.0)
    avgLoss += (loss - avgLoss) / float(batch+1)
    accuracy += (accVal - accuracy) / float(batch+1)
    model.variables = t.optim(params)
  return (avgLoss, accuracy)

proc getAccuracy*[T: ElemType](t: var Trainer, model: var Module, loader: Dataloader): float =
  ## Calculate the accuracy from the test data set.
  ## T should be the type of data returned from the loader.
  var data = newTensor[T](loader.shape)
  var targets = newTensor[int32](loader.batchSize)
  var accuracy = 0.0
  t.predict = newTensor[int32](0)
  for batch in getBatch(loader, data, targets):
    var params = initParams({"x": t.client.newBuffer(data), "y": t.client.newBuffer(targets)})
    model.setParams(params)
    t.tester.runWith(params)
    t.testAcc.runWith(params)
    let accVal = params["accuracy"].f32[].float
    if batch mod 10 == 0:
      debug &"test batch {batch}: accuracy = {accVal:.3f}"
      debug "  tgt: ", targets
      debug "  pred:", params["labels"].i32
    accuracy += (accVal - accuracy) / float(batch+1)
    t.predict = t.predict.append(params["labels"].i32)
  return accuracy

proc updateStats(t: var Trainer, entries: openarray[(string, float)]) =
  ## Append data to stats
  for (name, val) in entries:
    if t.stats.hasKey(name):
      t.stats[name].add val
    else:
      t.stats[name] = @[val]

proc statsPlots*(t: Trainer): JsonNode =
  ## Convert stats to format used by plots package
  result = %[]
  for i, plot in plotNames:
    let yaxis = if i == 0: "y" else: "y" & $(i+1)
    for trace in plot:
      result.add %*{"x": t.stats["epoch"], "y": t.stats[trace], "mode": "lines", "name": trace, "yaxis": yaxis}

proc getLayout*(epochs: int): JsonNode =
  result = plotLayout
  result["legend"] = %*{"x": 0.99, "xanchor": "right", "y": 0.99}
  result["xaxis"]["range"] = %*[1.0, epochs.float]

proc trainNetwork*[T: ElemType](t: var Trainer, model: var Module, train, test: DataLoader, epochs: int, plot = false) =
  ## Training run for given number of epochs
  t.stats = initTable[string, seq[float]]()
  var ws: WebSocket
  if plot:
    ws = openWebSocket()
  let start = getMonoTime().ticks
  for epoch in 1 .. epochs:
    let (loss, trainAcc) = trainEpoch[T](t, model, train)
    if loss.isNan:
      error "loss returns Nan value - aborting"
      break
    let testAcc = getAccuracy[T](t, model, test)
    echo &"epoch {epoch:3}:  loss: {loss:8.4f}  train accuracy: {trainAcc*100:.1f}%  test accuracy: {testAcc*100:.1f}%"
    t.updateStats({"epoch": epoch.float, "loss": loss, "train": trainAcc, "test": testAcc})
    if plot:
      updatePlot(ws, t.statsPlots, getLayout(epochs))
  let elapsed = float(getMonoTime().ticks - start) / 1e9
  echo &"elapsed time = {elapsed:.3f}s"
  if plot:
    ws.close

