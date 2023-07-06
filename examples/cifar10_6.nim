# simple CIFAR10 convolutional net with 6 x conv + 1 x linear layers
# params to the main proc can be set via cmd line argmuments
# output option will save predicted test labels to a file which can be read by imgview

{.warning[BareExcept]:off.}
import std/[strutils, strformat, math, logging, random, tables, monotimes]
import nimxla
import nimxla/[nn, data, train, image]
import cligen

setPrintOpts(precision=4, minWidth=8, floatMode=ffDecimal, threshold=100, edgeItems=5)


proc makeBlock(c: Client, rng: var Rand, id: string, nin, nout: int, dropoutRatio: float, dtype: DataType): Module =
  let conv1 = c.initConv2d(rng, id & ".1", nin, nout, kernelSize=3, padding=pad(1), biases=nil, dtype=dtype)
  let norm1 = c.initBatchNorm(rng, id & ".2", nout, dtype=dtype)
  let conv2 = c.initConv2d(rng, id & ".3", nout, nout, kernelSize=3, padding=pad(1), biases=nil, dtype=dtype)
  let norm2 = c.initBatchNorm(rng, id & ".4", nout, dtype=dtype)

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let l1 = conv1.forward(x, training, output)
    let l2 = norm1.forward(l1, training, output).relu
    let l3 = conv2.forward(l2, training, output)
    let l4 = norm2.forward(l3, training, output).relu
    l4.maxPool2d(2).dropout(dropoutRatio, training)

  result.add(conv1, norm1, conv2, norm2)


proc buildModel(c: Client, rng: var Rand, nclasses: int, mean, std: seq[float32], dtype: DataType): Module =
  let block1 = c.makeBlock(rng, "1", 3, 32, 0.2, dtype)
  let block2 = c.makeBlock(rng, "2", 32, 64, 0.3, dtype)
  let block3 = c.makeBlock(rng, "3", 64, 128, 0.4, dtype)
  let linear = c.initLinear(rng, "4", 2048, nclasses)

  result.forward = proc(x: Node, training: bool, output: var Outputs): Node =
    let b = x.builder
    let xf = x.convert(dtype) / b.constant(255, dtype)
    let mean = (b^mean).convert(dtype).reshape(1, 1, 1, 3)
    let std = (b^std).convert(dtype).reshape(1, 1, 1, 3)
    let xs = (xf - mean) / std
    let l1 = block1.forward(xs, training, output)
    let l2 = block2.forward(l1, training, output)
    let l3 = block3.forward(l2, training, output)
    linear.forward(l3.flatten(1).convert(F32), training, output).softmax

  result.info = "== cifar10_6 =="
  result.add(block1, block2, block3, linear)


proc main(epochs=80, learnRate=0.001, trainBatch=128, testBatch=250, seed=0i64, augment=true, 
          bfloat16=false, load="", checkpoint="data/cifar10_6", gpu=true, plot=false, debug=false) =
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
  var optim = c.optimAdam(model, learnRate)
  var t = Trainer(
    client:   c,
    model:    model,
    optim:    optim,
    sched:    newStepLR(optim, stepSize=30, gamma=0.25),
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
