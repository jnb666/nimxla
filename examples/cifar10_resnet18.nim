# CIFAR10 Resnet 18 convolutional net model
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

let defn = @[(64, 64, 1),   (64, 64, 1),   (64, 128, 2),  (128, 128, 1),
             (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]

proc makeBlock(c: Client, rng: var Rand, id: string, nin, nout, strides: int, dtype: DataType): Module =
  let conv1 = c.initConv2d(rng, id & ".1", nin, nout, kernelSize=3, strides=strides, padding=pad(1), biases=nil, dtype=dtype)
  let norm1 = c.initBatchNorm(rng, id & ".2", nout, dtype=dtype)
  let conv2 = c.initConv2d(rng, id & ".3", nout, nout, kernelSize=3, padding=pad(1), biases=nil, dtype=dtype)
  let norm2 = c.initBatchNorm(rng, id & ".4", nout, dtype=dtype)
  result.add(conv1, norm1, conv2, norm2)
  if strides == 1:
    result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
      let l1 = norm1.forward(conv1.forward(x, training, output), training, output).relu
      let l2 = norm2.forward(conv2.forward(l1, training, output), training, output)
      (l2 + x).relu
  else:
    let conv3 = c.initConv2d(rng, id & ".5", nin, nout, kernelSize=1, strides=strides, biases=nil, dtype=dtype)
    let norm3 = c.initBatchNorm(rng, id & ".6", nout, dtype=dtype)
    result.add(conv3, norm3)
    result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
      let l1 = norm1.forward(conv1.forward(x, training, output), training, output).relu
      let l2 = norm2.forward(conv2.forward(l1, training, output), training, output)
      let l3 = norm3.forward(conv3.forward(x, training, output), training, output)
      (l2 + l3).relu

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  result.info = "== cifar10_resnet18 =="
  let conv = c.initConv2d(rng, "1.1", 3, 64, kernelSize=3, padding=pad(1), biases=nil, dtype=dtype)
  let norm = c.initBatchNorm(rng, "1.2", 64, dtype=dtype)
  result.add(conv, norm)
  var blocks: seq[Module]
  for i, (nin, nout, strides) in defn:
    blocks.add c.makeBlock(rng, $(i+2), nin, nout, strides, dtype)
  result.add blocks
  let linear = c.initLinear(rng, "10", 512, nclasses)
  result.add linear

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(dtype) / b.constant(255, dtype)
    let mean = (b^mean).convert(dtype).reshape(1, 1, 1, 3)
    let std = (b^std).convert(dtype).reshape(1, 1, 1, 3)
    var xs = (xf - mean) / std
    xs = norm.forward(conv.forward(xs, training, output), training, output).relu
    for b in blocks:
      xs = b.forward(xs, training, output)
    linear.forward(xs.avgPool2d(4).flatten(1).convert(F32), training, output)


proc main(epochs=150, learnRate=0.02, wdecay=0.001, trainBatch=128, testBatch=250, seed=0i64, augment=true,
          bfloat16=true, load="", checkpoint="data/cifar10_res18", gpu=true, plot=false, debug=false) =
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
