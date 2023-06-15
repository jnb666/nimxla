# 2 layer MNIST MLP model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

import std/[strutils, math, logging, random, tables]
import nimxla
import nimxla/[nn, data, train]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, imgSize, nclasses: int): Module =
  let layer1 = c.initLinear("1", imgSize, 128)
  let layer2 = c.initLinear("2", 128, nclasses)
  result.forward = proc(x: Node, training: bool): Node =
    let b = x.builder
    let xf = x.convert(F32) / b^255f32
    let l1 = layer1.forward(xf.flatten(1)).relu
    layer2.forward(l1).softmax
  result.info = "== mnist_mlp =="
  result.add(layer1, layer2)

proc main(epochs = 10, learnRate = 0.01, trainBatch = 500, testBatch = 1000, seed: int64 = 0, 
          output = "", gpu = true, plot = false, debug = false) =
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
  var model = c.buildModel(prod(train.dataset.shape), nclasses)
  echo model
  # compile train funcs
  var t = Trainer(
    client:   c,
    optim:    c.optimAdam(model, learnRate),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(train.batchSize, nclasses),
    testAcc:  c.accuracyFunc(test.batchSize, nclasses)
  )
  echo "training with learning rate = ", learnRate
  trainNetwork[uint8](t, model, train, test, epochs, plot)
  if output != "":
    echo "saving test predictions to ", output
    t.predict.write(output)

dispatch main
