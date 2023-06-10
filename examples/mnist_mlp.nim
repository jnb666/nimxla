# 2 layer MNIST MLP model
import std/[strutils, strformat, math, sugar, logging, random, tables]
import nimxla
import nimxla/nn
import nimxla/data
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

const nclasses = 10

proc newModel(c: Client, imgSize, batchSize: int): (Module, Executable) =
  ## compile network either in test or train mode
  let b = newBuilder("mnist_mlp")
  let layer1 = c.initLinear(b, "1", nin=imgSize, nout=128, weights = normalInit(stddev=0.1), biases = constantInit(0f32))
  let layer2 = c.initLinear(b, "2", nin=128, nout=nclasses, weights = normalInit(stddev=0.1), biases = constantInit(0f32))

  var model = Module(builder: b)
  model.add(layer1, layer2)
  model.forward = proc(x: Node): Node =
    let xf = x.convert(F32) / b^255f32
    let l1 = layer1.forward(xf).relu
    result = layer2.forward(l1).softmax

  let x = b.parameter(U8, [batchSize, imgSize], "x")
  let y = b.parameter(I32, [batchSize], "y")
  let exec = c.compileTrain(model, x, yp => crossEntropyLoss(yp, y))
  return (model, exec)


proc calcAccuracy(c: Client, batch, nout: int): Executable =
  ## compile function to calculate the accuracy from the predicted output
  let b = newBuilder("accuracy")
  let predict = b.parameter(F32, [batch, nout], "predict")
  let labels = b.parameter(I32, [batch], "labels")
  let batchSize = b.constant(batch.float32)
  let comp = b.build sum(convert(predict.argmax(axis=1, ixType=I32) == labels, F32)) / batchSize
  c.compile(comp)


proc trainEpoch(c: Client, model: var Module, exec: Executable, loader: DataLoader, 
                optim: Optimizer, accFn: Executable): (float, float) =
  ## train on one epoch of batches of data from the training set, returns average loss and accuracy on training dara
  var images = newTensor[uint8](loader.batchSize, prod(loader.dataset.shape))
  var labels = newTensor[int32](loader.batchSize)
  var avgLoss = 0.0
  var accuracy = 0.0
  for batch in getBatch(loader, images, labels):
    var params = initParams({"x": c.newBuffer(images), "y": c.newBuffer(labels)})
    model.setParams(params)
    exec.runWith(params)
    if batch == 0:
      debug "train pred: ", params["pred"].f32
    let loss = params["loss"].f32[].float
    if loss.isNan:
      return (loss, 0.0)
    avgLoss += (loss - avgLoss) / float(batch+1)
    let acc = accFn.run([params["pred"], params["y"]]).f32[]
    accuracy += (acc - accuracy) / float(batch+1)
    debug &"train batch {batch:4}: loss = {loss:8.4f}  accuracy = {acc:8.4f}"
    model.variables = optim(params)
  return (avgLoss, accuracy)


proc getAccuracy(c: Client, model: var Module, exec: Executable, loader: Dataloader, accFn: Executable): float =
  ## calculate accuracy from test dataset
  var images = newTensor[uint8](loader.batchSize, prod(loader.dataset.shape))
  var labels = newTensor[int32](loader.batchSize)
  var accuracy = 0.0
  for batch in getBatch(loader, images, labels):
    var params = initParams({"x": c.newBuffer(images), "y": c.newBuffer(labels)})
    model.setParams(params)
    exec.runWith(params)
    if batch == 0:
      debug "test labels: ", labels
      debug "test pred: ", params["pred"].f32
    let acc = accFn.run([params["pred"], params["y"]]).f32[]
    accuracy += (acc - accuracy) / float(batch+1)
    debug &"test batch {batch:4}: accuracy = {acc:.3f}"
  return accuracy


proc main(epochs = 20, learnRate = 0.75, batchSize = 500, seed: int64 = 0, gpu = false, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = if gpu: newGPUClient() else: newCPUClient()
  info c
  if seed != 0:
    randomize(seed)
  else:
    randomize()
  # get data
  let trainData = initLoader(mnistDataset(train = true), batchSize, shuffle = true)
  info trainData
  let testData = initLoader(mnistDataset(train = false), batchSize)
  info testData
  let imgSize = prod(trainData.dataset.shape)
  # compile model executable
  var (model, exec) = c.newModel(imgSize, batchSize)
  for p in model.variables:
    debug &"initial {p.name}: {p.data.f32}"
  # compile optimizer used to update the weights and function to calc the accuracy on the test set
  let optim = c.optimSGD(model, learnRate)
  let accFn = c.calcAccuracy(batchSize, nclasses)

  info "training with learning rate = ", learnRate
  for epoch in 1 .. epochs:
    let (loss, trainAcc) = c.trainEpoch(model, exec, trainData, optim, accFn)
    if loss.isNan:
      error "loss returns Nan value - aborting"
      break
    let testAcc = c.getAccuracy(model, exec, testData, accFn)
    info &"epoch {epoch:3}:  loss: {loss:8.4f}  train accuracy: {trainAcc*100:.1f}%  test accuracy: {testAcc*100:.1f}%"

dispatch main