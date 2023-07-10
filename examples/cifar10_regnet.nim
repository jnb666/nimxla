# CIFAR10 RegnetX200 / X400 convolutional net model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

let defn_200 = @[(1,  24, 1, 8),
                 (1,  56, 1, 8),
                 (4, 152, 2, 8),
                 (7, 368, 2, 8)]

let defn_400 = @[(1,   32, 1, 16),
                 (2,   64, 1, 16),
                 (7,  160, 2, 16),
                 (12, 384, 2, 16)]
let
  convInit = heInit(fanOut = true)
  linearInit = normalInit(stddev = 0.001)


proc makeBlock(c: Client, rng: var Rand, id: string, nin, nout, strides, groups: int, dtype: DataType): Module =
  let conv1 = c.initConv2dBatchNorm(rng, id & ".a1", nin, nout, kernelSize=1, weights=convInit, dtype=dtype)
  let conv2 = c.initConv2dBatchNorm(rng, id & ".a2", nout, nout, kernelSize=3, strides=strides, padding=pad(1),
                                                     groups=(nout div groups), weights=convInit, dtype=dtype)
  let conv3 = c.initConv2dBatchNorm(rng, id & ".a3", nout, nout, kernelSize=1, weights=convInit, dtype=dtype)
  result.info = "residual:"
  result.add(conv1, conv2, conv3)
  var shortcut: Module
  if strides != 1 or nin != nout:
    shortcut = c.initConv2dBatchNorm(rng, id & ".b1", nin, nout, kernelSize=1, strides=strides,
                                                      weights=convInit, dtype=dtype)
    result.add shortcut
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let l1 = conv1.forward(x, training, output).relu
    let l2 = conv2.forward(l1, training, output).relu
    result = conv3.forward(l2, training, output)
    if strides != 1 or nin != nout:
      result = (result + shortcut.forward(x, training, output)).relu
    else:
      result = (result + x).relu

proc buildModel(c: Client, rng: var Rand, typ: string, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  result.info = "== cifar10_regnet" & typ & " =="
  let conv1 = c.initConv2dBatchNorm(rng, "1", 3, 64, kernelSize=3, padding=pad(1), weights=convInit, dtype=dtype)
  result.add conv1
  var blocks: seq[Module]
  var nin = 64
  let defn = if typ == "X400":
    defn_400
  else:
    defn_200
  for i, (depth, nout, strides, groups) in defn:
    var stride = strides
    for j in 1..depth:
      blocks.add c.makeBlock(rng, &"{i+2}.{j}", nin, nout, stride, groups, dtype)
      nin = nout
      stride = 1
  result.add blocks
  let linear = c.initLinear(rng, "6", nin, nclasses, weights=linearInit)
  result.add linear

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(dtype) / b.constant(255, dtype)
    let mean = (b^mean).convert(dtype).reshape(1, 1, 1, 3)
    let std = (b^std).convert(dtype).reshape(1, 1, 1, 3)
    var xs = (xf - mean) / std
    xs = conv1.forward(xs, training, output).relu
    for b in blocks:
      xs = b.forward(xs, training, output)
    linear.forward(xs.avgPool2d(8).flatten(1).convert(F32), training, output)


proc main(epochs=100, learnRate=0.1, wdecay=0.0005, trainBatch=128, testBatch=250, seed=0i64, augment=true,
          bfloat16=true, load="", x400 = false, checkpoint="data/cifar10_regnet", gpu=true, plot=false, debug=false) =
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
  let typ = if x400: "X400" else: "X200"
  var model = c.buildModel(rng, typ, nclasses, mean, std, dtype)
  echo model
  # compile train funcs
  var t = Trainer(
    client:   c,
    model:    model,
    optim:    c.optimSGD(model, learnRate, wdecay, momentum=0.9, nesterov=true),
    trainer:  c.trainFunc(model, U8, train.shape, crossEntropyLoss),
    tester:   c.testFunc(model, U8, test.shape),
    trainAcc: c.accuracyFunc(train.batchSize, nclasses),
    testAcc:  c.accuracyFunc(test.batchSize, nclasses)
  )
  t.sched = newChainedScheduler(
    newLinearLR(t.optim, epochs=20, startFactor=0.1, endFactor=1.0),
    newCosineAnnealingLR(t.optim, tMax=epochs-20, lrMin=0.001)
  )
  echo "optimizer: ", t.sched
  if load != "":
    t.loadCheckpoint(load)
  trainNetwork[uint8](t, train, test, epochs, plot, checkpoint & typ)
  train.shutdown()

dispatch main
