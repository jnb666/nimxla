# CIFAR10 MobileNet v2 convolutional net model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

let defn = @[(1,  16, 1, 1), 
             (6,  24, 2, 1),
             (6,  32, 3, 2),
             (6,  64, 4, 2),
             (6,  96, 3, 1),
             (6, 160, 3, 2),
             (6, 320, 1, 1)]

let
  convInit = heInit(fanOut = true)
  linearInit = normalInit(stddev = 0.001)

proc makeBlock(c: Client, rng: var Rand, id: string, nin, nout, expand, strides: int, dtype: DataType): Module =
  let d = nin*expand
  let conv1 = c.initConv2dBatchNorm(rng, id & ".a1", nin, d, kernelSize=1, weights=convInit, dtype=dtype)
  let conv2 = c.initConv2dBatchNorm(rng, id & ".a2", d, d, kernelSize=3, strides=strides, padding=pad(1), 
                                                     groups=d, weights=convInit, dtype=dtype)
  let conv3 = c.initConv2dBatchNorm(rng, id & ".a3", d, nout, kernelSize=1, weights=convInit, dtype=dtype)
  result.info = if strides == 1: "residual:" else: "block:"
  result.add(conv1, conv2, conv3)
  var shortcut: Module
  if strides == 1 and nin != nout:
    shortcut = c.initConv2dBatchNorm(rng, id & ".b1", nin, nout, kernelSize=1, weights=convInit, dtype=dtype)
    result.add shortcut
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let l1 = conv1.forward(x, training, output).relu
    let l2 = conv2.forward(l1, training, output).relu
    result = conv3.forward(l2, training, output)
    if strides == 1:
      if nin != nout:
        result = result + shortcut.forward(x, training, output)
      else:
        result = result + x

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  result.info = "== cifar10_mobilenet2 =="
  let conv1 = c.initConv2dBatchNorm(rng, "1", 3, 32, kernelSize=3, padding=pad(1), weights=convInit, dtype=dtype)
  result.add conv1
  var blocks: seq[Module]
  var nin = 32
  for i, (expand, nout, layers, strides) in defn:
    var stride = strides
    for j in 1..layers:
      blocks.add c.makeBlock(rng, &"{i+2}.{j}", nin, nout, expand, stride, dtype)
      nin = nout
      stride = 1
  result.add blocks
  let conv2 = c.initConv2dBatchNorm(rng, "9", nin, 1280, kernelSize=1, weights=convInit, dtype=dtype)
  let linear = c.initLinear(rng, "10", 1280, nclasses, weights=linearInit)
  result.add(conv2, linear)

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(dtype) / b.constant(255, dtype)
    let mean = (b^mean).convert(dtype).reshape(1, 1, 1, 3)
    let std = (b^std).convert(dtype).reshape(1, 1, 1, 3)
    var xs = (xf - mean) / std
    xs = conv1.forward(xs, training, output).relu
    for b in blocks:
      xs = b.forward(xs, training, output)
    xs = conv2.forward(xs, training, output).relu
    linear.forward(xs.avgPool2d(4).flatten(1).convert(F32), training, output)


proc main(epochs=150, learnRate=0.05, wdecay=0.001, trainBatch=128, testBatch=250, seed=0i64, augment=true,
          bfloat16=true, load="", checkpoint="data/cifar10_mobile2", gpu=true, plot=false, debug=false) =
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
