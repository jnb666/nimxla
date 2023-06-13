## The train module has common functions for training neural networks.
## It depends on the nn and data modules in addition to the core nimxla modules.

import std/[tables, logging, math, strformat, sugar]
import ../nimxla
import nn, data


proc trainFunc*(c: Client, model: Module, dtype: DataType, shape: seq[int], lossFn: proc(pred, target: Node): Node): Executable =
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
  ## - array predictions of shape <outType>[batch, nout]
  ## - array correct labels of shape <labelType>[batch]
  ## The maximum index in each row from the predictions is compared with the correspoding label
  ## and summed to return the accuracy as a float32 scalar between 0 and 1.
  debug &"calcAccuracy batch={batch} nout={nout}"
  let b = newBuilder("accuracy")
  let predict = b.parameter(outType, [batch, nout], "predict")
  let labels = b.parameter(labelType, [batch], "labels")
  let avg = mean(convert(predict.argmax(axis=1, ixType=labelType) == labels, outType))
  let comp = b.build(avg.convert(F32))
  c.compile(comp)

proc trainEpoch*[T: ElemType](c: Client, model: var Module, exec: Executable, loader: DataLoader, 
                optim: Optimizer, accFn: Executable): (float, float) =
  ## Train on one epoch of batches of data from the training set, returns average loss and accuracy on training dara
  ## T should be the type of data returned from the loader. Output loss should be a float32 scalar.
  var data = newTensor[T](loader.shape)
  var labels = newTensor[int32](loader.batchSize)
  var avgLoss = 0.0
  var accuracy = 0.0
  for batch in getBatch(loader, data, labels):
    var params = initParams({"x": c.newBuffer(data), "y": c.newBuffer(labels)})
    model.setParams(params)
    exec.runWith(params)
    if batch == 0:
      debug "train pred: ", params["pred"].toLiteral
    let loss = params["loss"].f32[].float
    if loss.isNan:
      return (loss, 0.0)
    avgLoss += (loss - avgLoss) / float(batch+1)
    let acc = accFn.run([params["pred"], params["y"]]).f32[]
    accuracy += (acc - accuracy) / float(batch+1)
    model.variables = optim(params)
  return (avgLoss, accuracy)

proc getAccuracy*[T: ElemType](c: Client, model: var Module, exec: Executable, loader: Dataloader, accFn: Executable): float =
  ## Calculate the accuracy from the test data set.
  ## T should be the type of data returned from the loader.
  var data = newTensor[T](loader.shape)
  var labels = newTensor[int32](loader.batchSize)
  var accuracy = 0.0
  for batch in getBatch(loader, data, labels):
    var params = initParams({"x": c.newBuffer(data), "y": c.newBuffer(labels)})
    model.setParams(params)
    exec.runWith(params)
    if batch == 0:
      debug "test labels: ", labels
      debug "test pred: ", params["pred"].toLiteral
    let acc = accFn.run([params["pred"], params["y"]]).f32[]
    accuracy += (acc - accuracy) / float(batch+1)
  return accuracy


