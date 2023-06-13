# simple MNIST convolutional net with 2 x conv + 2 x linear layers 
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, nclasses: int): Module =
  let conv1 = c.initConv2d("1", 1,  20, kernelSize=5, biases=nil)
  let conv2 = c.initConv2d("2", 20, 40, kernelSize=5)
  let linear1 = c.initLinear("3", 640, 100)
  let linear2 = c.initLinear("4", 100, nclasses)
  result.forward = proc(x: Node): Node =
    let b = x.builder
    let xf = x.convert(F32) / b^255f32
    let l1 = conv1.forward(xf).relu.maxPool2d(2)
    let l2 = conv2.forward(l1).relu.maxPool2d(2)
    let l3 = linear1.forward(l2.flatten(1)).relu
    linear2.forward(l3).softmax
  result.info = "== mnist_cnn =="
  result.add(conv1, conv2, linear1, linear2)  

proc main(epochs = 10, learnRate = 0.01, trainBatch = 500, testBatch = 1000, seed: int64 = 0, gpu = true, printWeights = false, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = newClient(useGPU = gpu)
  echo c
  if seed != 0: randomize(seed) else: randomize()
  # get data
  let train = initLoader(mnistDataset(train = true), trainBatch, shuffle = true)
  echo train
  let test = initLoader(mnistDataset(train = false), testBatch)
  echo test
  let nclasses = train.dataset.classes.len
  # build model
  var model = c.buildModel(nclasses)
  echo model
  if printWeights:
    for p in model.variables:
      echo &"initial {p.name}: {p.data.f32}"
  let trainer = c.trainFunc(model, U8, train.shape, crossEntropyLoss)
  let tester = c.testFunc(model, U8, test.shape)
  # compile optimizer used to update the weights and function to calc the accuracy
  let optim = c.optimAdam(model, learnRate)
  let aTrain = c.accuracyFunc(train.batchSize, nclasses)
  let aTest = c.accuracyFunc(test.batchSize, nclasses)

  echo "training with learning rate = ", learnRate
  let start = getMonoTime().ticks
  for epoch in 1 .. epochs:
    let (loss, trainAcc) = trainEpoch[uint8](c, model, trainer, train, optim, aTrain)
    if loss.isNan:
      error "loss returns Nan value - aborting"
      break
    let testAcc = getAccuracy[uint8](c, model, tester, test, aTest)
    echo &"epoch {epoch:3}:  loss: {loss:8.4f}  train accuracy: {trainAcc*100:.1f}%  test accuracy: {testAcc*100:.1f}%"
  let elapsed = float(getMonoTime().ticks - start) / 1e9

  echo &"elapsed time = {elapsed:.3f}s"
  if printWeights:
    for p in model.variables:
      echo &"final {p.name}: {p.data.f32}"

dispatch main