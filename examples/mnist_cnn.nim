# simple MNIST convolutional net with 2 x conv + 2 x linear layers 
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, math, random, logging, tables]
import nimxla
import nimxla/[nn, data, train]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, rng: var Rand, nclasses: int): Module =
  let conv1 = c.initConv2d(rng, "1", 1,  20, kernelSize=5, biases=nil)
  let conv2 = c.initConv2d(rng, "2", 20, 40, kernelSize=5)
  let linear1 = c.initLinear(rng, "3", 640, 100)
  let linear2 = c.initLinear(rng, "4", 100, nclasses)
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(F32) / b^255f32
    let l1 = conv1.forward(xf, training, output).relu.maxPool2d(2)
    let l2 = conv2.forward(l1, training, output).relu.maxPool2d(2)
    let l3 = linear1.forward(l2.flatten(1), training, output).relu
    linear2.forward(l3, training, output).softmax
  result.info = "== mnist_cnn =="
  result.add(conv1, conv2, linear1, linear2)  

proc main(epochs = 10, learnRate = 0.01, trainBatch = 500, testBatch = 1000, seed: int64 = 0, 
          output = "", gpu = true, plot = false, debug = false) =
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
  var model = c.buildModel(rng, nclasses)
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
  echo "optimizer: ", t.optim
  trainNetwork[uint8](t, model, train, test, epochs, plot=plot)
  if output != "":
    echo "saving test predictions to ", output
    t.predict.write(output)

dispatch main
