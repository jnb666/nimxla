# CIFAR10 VGG convolutional net model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

proc stack(c: Client, rng: var Rand, id: string, n, nin, nout: int, dtype: DataType): Module =
  var layers: seq[Module]
  for i in 1..n:
    layers.add c.initConv2dBatchNorm(rng, id & "." & $i, if i == 1: nin else: nout, nout,
                                     kernelSize=3, padding=pad(1), dtype=dtype)
  result.add layers
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    var xv = x
    for l in layers:
      xv = l.forward(xv, training, output).relu
    xv.maxPool2d(2)

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  result.info = "== cifar10_vgg =="
  var blocks: seq[Module]
  blocks.add c.stack(rng, "1", 2, 3, 64, dtype)
  blocks.add c.stack(rng, "2", 2, 64, 128, dtype)
  blocks.add c.stack(rng, "3", 3, 128, 256, dtype)
  blocks.add c.stack(rng, "4", 3, 256, 512, dtype)
  blocks.add c.stack(rng, "5", 3, 512, 512, dtype)
  result.add blocks
  let linear = c.initLinear(rng, "6", 512, nclasses)
  result.add linear

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(dtype) / b.constant(255, dtype)
    let mean = (b^mean).convert(dtype).reshape(1, 1, 1, 3)
    let std = (b^std).convert(dtype).reshape(1, 1, 1, 3)
    var xs = (xf - mean) / std
    for b in blocks:
      xs = b.forward(xs, training, output)
    linear.forward(xs.flatten(1).convert(F32), training, output)


proc main(epochs=100, learnRate=0.01, wdecay=0.005, trainBatch=128, testBatch=250, seed=0i64, augment=true,
          bfloat16=true, load="", checkpoint="data/cifar10_vgg", gpu=true, plot=false, debug=false) =
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
  let dtype = if bfloat16: BF16 else: F32
  var model = c.buildModel(rng, nclasses, mean, std, dtype)
  echo model
  # compile train funcs
  var optim = c.optimSGD(model, learnRate, wdecay, momentum=0.9, nesterov=true)
  var t = Trainer(
    client:   c,
    model:    model,
    optim:    optim,
    sched:    newCosineAnnealingLR(optim, tMax=epochs),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(train.batchSize, nclasses),
    testAcc:  c.accuracyFunc(test.batchSize, nclasses)
  )
  echo "optimizer: ", t.sched
  if load != "":
    t.loadCheckpoint(load)
  trainNetwork[uint8](t, train, test, epochs, plot, checkpoint)
  train.shutdown()

dispatch main
