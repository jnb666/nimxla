## The train module has common functions for training neural networks.
## It depends on the nn and data modules in addition to the core nimxla modules.

import std/[tables, json, logging, math, strformat, strutils, sugar, times, monotimes]
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
    heatmap*:  Tensor[int32]

let
  plotNames = @[@["loss"], @["train", "test"]]

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

proc trainEpoch*[T: ElemType](t: var Trainer, model: var Module, loader: var DataLoader): (float, float) =
  ## Train on one epoch of batches of data from the training set, returns average loss and accuracy on training dara
  ## T should be the type of data returned from the loader. Output loss should be a float32 scalar.
  var data = newTensor[T](loader.shape)
  var targets = newTensor[int32](loader.batchSize)
  var avgLoss = 0.0
  var accuracy = 0.0
  for batch in getBatch(loader, data, targets):
    var params = initParams({"x": t.client.newBuffer(data), "y": t.client.newBuffer(targets)})
    model.getParams(params)
    t.trainer.runWith(params)
    model.update(params)
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
    model.setParams(t.optim(params))
  return (avgLoss, accuracy)

proc getAccuracy*[T: ElemType](t: var Trainer, model: var Module, loader: var Dataloader): float =
  ## Calculate the accuracy from the test data set.
  ## T should be the type of data returned from the loader.
  var data = newTensor[T](loader.shape)
  var targets = newTensor[int32](loader.batchSize)
  var accuracy = 0.0
  t.predict = newTensor[int32](0)
  let nclasses = loader.dataset.classes.len
  t.heatmap = zeros[int32](nclasses, nclasses)
  for batch in getBatch(loader, data, targets):
    var params = initParams({"x": t.client.newBuffer(data), "y": t.client.newBuffer(targets)})
    model.getParams(params)
    t.tester.runWith(params)
    t.testAcc.runWith(params)
    let accVal = params["accuracy"].f32[].float
    if batch mod 10 == 0:
      debug &"test batch {batch}: accuracy = {accVal:.3f}"
      debug "  tgt: ", targets
      debug "  pred:", params["labels"].i32
    accuracy += (accVal - accuracy) / float(batch+1)
    let labels = params["labels"].i32
    t.predict = t.predict.append(labels)
    for i in 0 ..< targets.len:
      let (x, y) = (targets[i].int, labels[i].int)
      t.heatmap[y, x] = t.heatmap[y, x] + 1
  return accuracy

proc updateStats(t: var Trainer, entries: openarray[(string, float)]) =
  ## Append data to stats
  for (name, val) in entries:
    if t.stats.hasKey(name):
      t.stats[name].add val
    else:
      t.stats[name] = @[val]

proc statsPlots*(t: Trainer, classes: seq[string]): JsonNode =
  ## Convert stats to format used by plots package
  result = %[]
  for i, plot in plotNames:
    for trace in plot:
      result.add %*{"x": t.stats["epoch"], "y": t.stats[trace], "mode": "lines", "name": trace, "yaxis": "y" & $(i+1)}
  # heatmap plot
  let nclasses = classes.len
  var heatmap: seq[seq[float]]
  let scale = nclasses / t.predict.len
  for y in 0 ..< nclasses:
    var row: seq[float]
    for x in 0 ..< nclasses:
      row.add t.heatmap[y, x].float * scale
    heatmap.add row
    result.add %*{"z": heatmap, "x": classes, "y": classes, "type": "heatmap", "xaxis": "x3", "yaxis": "y3", 
                  "colorscale": "Blackbody", "showscale": false}

proc getLayout*(epochs: int): JsonNode =
  %*{
    "xaxis":  {"domain": [0.0, 0.49], "anchor": "y2", "title": "epoch", "range": [1.0, epochs.float]},
    "yaxis":  {"domain": [0.505, 1.0], "anchor": "x1", "title": "loss"},
    "yaxis2": {"domain": [0.0, 0.495], "anchor": "x2", "title": "accuracy"},
    "xaxis3": {"domain": [0.525, 1.0], "anchor": "y3"},
    "yaxis3": {"domain": [0.35, 1.0], "anchor": "x3"},
    "legend": {"x": 0.48, "xanchor": "right", "y": 0.99},
    "margin": {"t": 30, "l": 60, "r": 0, "b": 30},
  }

proc format(d: Duration): string =
  let secs = d.inSeconds
  let dp = toParts(d)
  if secs == 0:
    &"{dp[Milliseconds]}ms"
  elif secs < 60:
    &"{dp[Seconds]}.{dp[Milliseconds]}s"
  elif secs < 3600:
    &"{dp[Minutes]}m:{dp[Seconds]}.{dp[Milliseconds]}s"
  else:
    &"{secs div 3600}h:{dp[Minutes]}m:{dp[Seconds]}s"


proc trainNetwork*[T: ElemType](t: var Trainer, model: var Module, train, test: var DataLoader, epochs: int, plot = false) =
  ## Training run for given number of epochs. If transform is set it will be applied to each batch of training data.
  t.stats = initTable[string, seq[float]]()
  var ws: WebSocket
  if plot:
    ws = openWebSocket()
  let start = getMonoTime()
  for epoch in 1 .. epochs:
    let startEpoch = getMonoTime()
    let (loss, trainAcc) = trainEpoch[T](t, model, train)
    if loss.isNan:
      error "loss returns Nan value - aborting"
      break
    let testAcc = getAccuracy[T](t, model, test)
    t.updateStats({"epoch": epoch.float, "loss": loss, "train": trainAcc, "test": testAcc})
    if plot:
      updatePlot(ws, t.statsPlots(test.dataset.classes), getLayout(epochs))
    let elapsed = format(getMonoTime() - startEpoch)
    echo &"epoch {epoch:3}:  loss: {loss:8.4f}  train accuracy: {trainAcc*100:.1f}%  test accuracy: {testAcc*100:.1f}%  elapsed: {elapsed}"

  let totalElapsed = format(getMonoTime() - start)
  echo &"total elapsed time = {totalElapsed}"
  if plot:
    ws.close

