# simple CIFAR10 convolutional net with 4 x conv + 2 x linear layers 
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32]): Module =
  let conv1 = c.initConv2d(rng, "1", 3,  32, kernelSize=3, padding=pad(1))
  let conv2 = c.initConv2d(rng, "2", 32, 32, kernelSize=3)
  let conv3 = c.initConv2d(rng, "3", 32, 64, kernelSize=3, padding=pad(1))
  let conv4 = c.initConv2d(rng, "4", 64, 64, kernelSize=3)
  let linear1 = c.initLinear(rng, "5", 2304, 512)
  let linear2 = c.initLinear(rng, "6", 512, nclasses)
  result.forward = proc(x: Node, training: bool): Node =
    let b = x.builder
    let xf = (x.convert(F32) / b^255f32) 
    let xs = (xf - reshape(b^mean, [1, 1, 1, 3])) / reshape(b^std, [1, 1, 1, 3])
    let l1 = conv1.forward(xs).relu
    let l2 = conv2.forward(l1).relu.maxPool2d(2).dropout(0.25, training)
    let l3 = conv3.forward(l2).relu
    let l4 = conv4.forward(l3).relu.maxPool2d(2).dropout(0.25, training)
    let l5 = linear1.forward(l4.flatten(1)).relu.dropout(0.5, training)
    linear2.forward(l5).softmax
  result.info = "== cifar10_4 =="
  result.add(conv1, conv2, conv3, conv4, linear1, linear2)  

proc main(epochs = 50, learnRate = 0.001, trainBatch = 200, testBatch = 500, seed: int64 = 0,
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
    transforms = @[randomFlip(Horizontal), randomWrap(4, 4)]
  var train = newLoader(rng, trainBatch, shuffle = true)
  train.start(cifar10Dataset(train = true), channels=3, transforms)
  echo train
  var test = newLoader(rng, testBatch)
  test.start(cifar10Dataset(train = false), channels=3)
  echo test
  let nclasses = train.dataset.classes.len
  let (mean, std) = train.dataset.normalization
  # build model
  var model = c.buildModel(rng, nclasses, mean, std)
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
