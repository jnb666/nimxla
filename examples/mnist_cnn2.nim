# simple MNIST convolutional net with 2 x conv + 2 x linear layers 
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: float32): Module =
  let conv1 = c.initConv2d(rng, "1", 1,  32, kernelSize=5)
  let conv2 = c.initConv2d(rng, "2", 32, 64, kernelSize=5)
  let linear1 = c.initLinear(rng, "3", 1024, 1024)
  let linear2 = c.initLinear(rng, "4", 1024, nclasses)
  result.forward = proc(x: Node, training: bool): Node =
    let b = x.builder
    let xf = (x.convert(F32) / b^255f32) 
    let xs = (xf - b^mean) / b^std
    let l1 = conv1.forward(xs).maxPool2d(2)
    let l2 = conv2.forward(l1).maxPool2d(2)
    let l3 = linear1.forward(l2.flatten(1)).relu.dropout(0.5, training)
    linear2.forward(l3).softmax
  result.info = "== mnist_cnn2 =="
  result.add(conv1, conv2, linear1, linear2)  

proc main(epochs = 50, learnRate = 0.0002, trainBatch = 500, testBatch = 1000, seed: int64 = 0, 
          augment = true, output = "", gpu = true, plot = false, debug = false) =
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
  var t = Trainer(
    client:   c,
    optim:    c.optimAdam(model, learnRate),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(train.batchSize, nclasses),
    testAcc:  c.accuracyFunc(test.batchSize, nclasses)
  )
  echo "training with learning rate = ", learnRate
  trainNetwork[uint8](t, model, train, test, epochs, plot=plot)
  if output != "":
    echo "saving test predictions to ", output
    t.predict.write(output)
  train.shutdown()

dispatch main
