# 2 layer MNIST MLP model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, math, random, logging, tables]
import nimxla
import nimxla/[nn, data, train]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, rng: var Rand, imgSize, nclasses: int): Module =
  let layer1 = c.initLinear(rng, "1", imgSize, 128)
  let layer2 = c.initLinear(rng, "2", 128, nclasses)
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(F32) / b^255f32
    let l1 = layer1.forward(xf.flatten(1), training, output).relu
    layer2.forward(l1, training, output).softmax
  result.info = "== mnist_mlp =="
  result.add(layer1, layer2)

proc main(epochs = 10, learnRate = 0.01, trainBatch = 500, testBatch = 1000, seed = 0i64,
          gpu = true, plot = false, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = newClient(useGPU = gpu)
  echo c
  var rng = initRandom(seed)
  # get data
  var train = newLoader(rng, trainBatch, shuffle = true)
  train.start(mnistDataset(train = true), channels = 1)
  echo train
  var test = newLoader(rng, testBatch)
  test.start(mnistDataset(train = false), channels = 1)
  echo test
  let nclasses = train.dataset.classes.len
  # build model
  var model = c.buildModel(rng, prod(train.dataset.shape), nclasses)
  echo model
  # compile train funcs
  var t = Trainer(
    client:   c,
    model:    model,
    optim:    c.optimAdam(model, learnRate),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(train.batchSize, nclasses),
    testAcc:  c.accuracyFunc(test.batchSize, nclasses)
  )
  echo "optimizer: ", t.optim
  trainNetwork[uint8](t, train, test, epochs, plot=plot)

dispatch main
