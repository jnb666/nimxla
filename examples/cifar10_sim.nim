# CIFAR10 convolutional net with 15 x conv + 1 x linear layers
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)

let defn1 = @[
  (64, false),  (128, false), (128, false), (128, true),  (128, false), 
  (128, false), (256, true),  (256, false), (256, true),  (512, false)]

let defn2 = @[
  (2048, 1, false, true), (256, 1, true, true), (256, 3, true, false)
]

proc makeBlock1(c: Client, rng: var Rand, id: string, nin, nout: int, pooling: bool, dtype: DataType): Module =
  let conv = c.initConv2dBatchNorm(rng, id, nin, nout, kernelSize=3, padding=pad(1), dtype=dtype)
  result.add conv
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let l1 = conv.forward(x, training, output).relu
    if pooling:
      l1.maxPool2d(2).dropout(0.2, training)
    else:
      l1.dropout(0.2, training)

proc makeBlock2(c: Client, rng: var Rand, id: string, nin, nout, ksize: int, pooling, dropout: bool, dtype: DataType): Module =
  let pads = if ksize == 3: pad(1) else: pad(0)
  let conv = c.initConv2d(rng, id, nin, nout, kernelSize=ksize, padding=pads, dtype=dtype)
  result.add conv
  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let l1 = conv.forward(x, training, output).relu
    if pooling and dropout:
      l1.maxPool2d(2).dropout(0.2, training)
    elif pooling:
      l1.maxPool2d(2)
    elif dropout:
      l1.dropout(0.2, training)
    else:
      l1

proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  result.info = "== cifar10_sim =="
  var blocks: seq[Module]
  var nin = 3
  for i, (nout, pooling) in defn1:
    blocks.add c.makeBlock1(rng, "1." & $i, nin, nout, pooling, dtype)
    nin = nout
  for i, (nout, ksize, pooling, dropout) in defn2:
    blocks.add c.makeBlock2(rng, "2." & $i, nin, nout, ksize, pooling, dropout, dtype)
    nin = nout
  result.add blocks
  let linear = c.initLinear(rng, "3", 256, nclasses)
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


proc main(epochs=100, learnRate=0.001, wdecay=0.01, trainBatch=128, testBatch=250, seed=0i64, augment=true,
          bfloat16=true, load="", checkpoint="data/cifar10_sim", gpu=true, plot=false, debug=false) =
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
  var optim = c.optimAdamW(model, learnRate, wdecay)
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
