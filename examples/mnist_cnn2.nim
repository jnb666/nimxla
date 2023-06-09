# simple MNIST convolutional net with 2 x conv + 2 x linear layers 
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, math, random, logging, tables]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)


proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: float32): Module =
  let conv1 = c.initConv2d(rng, "1", 1,  32, kernelSize=5)
  let conv2 = c.initConv2d(rng, "2", 32, 64, kernelSize=5)
  let linear1 = c.initLinear(rng, "3", 1024, 1024)
  let linear2 = c.initLinear(rng, "4", 1024, nclasses)
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = (x.convert(F32) / b^255f32) 
    let xs = (xf - b^mean) / b^std
    let l1 = conv1.forward(xs, training, output).maxPool2d(2)
    let l2 = conv2.forward(l1, training, output).maxPool2d(2)
    let l3 = linear1.forward(l2.flatten(1), training, output).relu.dropout(0.5, training)
    linear2.forward(l3, training, output)
  result.info = "== mnist_cnn2 =="
  result.add(conv1, conv2, linear1, linear2)  


proc main(epochs=30, learnRate=0.001, wdecay=1e-5, trainBatch=200, testBatch=1000, seed=0i64, augment=true,
          load="", checkpoint="data/mnist_cnn2", gpu=true, plot=false, debug=false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  # init client
  let c = newClient(useGPU = gpu)
  echo c
  var rng = initRandom(seed)
  # get data
  var transforms: seq[ImageOp]
  if augment:
    transforms = @[randomAffine(rotate=15, p=0.25), randomAffine(scale=(0.85,1.15), p=0.25), randomElastic(scale=0.5, p=0.25)]
  var train = newLoader(rng, trainBatch, shuffle = true)
  train.start(mnistDataset(train = true), channels = 1, transforms)
  echo train
  var test = newLoader(rng, testBatch)
  test.start(mnistDataset(train = false), channels = 1)
  echo test
  let nclasses = train.dataset.classes.len
  let (mean, std) = train.dataset.normalization
  # build model
  var model = c.buildModel(rng, nclasses, mean[0], std[0])
  echo model
  # compile train funcs
  var optim = c.optimAdam(model, learnRate, wdecay)
  var t = Trainer(
    client:   c,
    model:    model,
    optim:    optim,
    sched:    newStepLR(optim, stepSize=10, gamma=0.2),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(trainBatch, nclasses),
    testAcc:  c.accuracyFunc(testBatch, nclasses)
  )
  echo "optimizer: ", t.sched
  if load != "":
    t.loadCheckpoint(load)
  trainNetwork[uint8](t, train, test, epochs, plot, checkpoint)
  train.shutdown()

dispatch main
