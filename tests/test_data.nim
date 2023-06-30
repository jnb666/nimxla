{.warning[BareExcept]:off.}
import std/[unittest, logging, strutils, strformat, random]
import nimxla
import nimxla/[data, image]

const debug {.booldefine.} = false


suite "data":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  var rng = initRand(0)

  test "mnist":
    setPrintOpts(minWidth = 4)
    let d = mnistDataset(train = false)
    debug "classes = ", d.classes.join(", ")
    check d.len == 10000
    check d.shape == [28, 28, 1]
    check d.classes.len == 10
    var img = newSeq[uint8](28*28)
    let label = d.getItem(0, img[0].addr)
    let t = img.toTensor.reshape(d.shape[0..^2])
    debug t
    debug label
    check label == 7
    setPrintOpts()

  test "cifar10":
    let d = cifar10Dataset(train = true)
    debug "classes = ", d.classes.join(", ")
    check d.len == 50000
    check d.shape == [32, 32, 3]
    check d.classes.len == 10
    
  test "loader":
    setPrintOpts(minWidth = 4)
    let batch = 20
    var loader = newLoader(rng, batch)
    loader.start(mnistDataset(train = true), channels = 1)
    debug loader
    var data = newTensor[uint8](batch, 28, 28)
    var labels = newTensor[int32](batch)
    var count = 0
    for i in getBatch(loader, data, labels):
      if i == 42:
        debug data.at(19)
        debug labels        
      count += 1
    debug "count = ", count
    check count == 60000 div batch
    setPrintOpts()

  test "transform":
    var loader = newLoader(rng, batchSize=100)
    loader.start(cifar10Dataset(train = false), channels = 3, randomFlip(Horizontal), randomWrap(4, 4))
    debug loader
    var data = newTensor[uint8](100, 32, 32, 3)
    var labels = newTensor[int32](100)
    var count = 0
    for i in getBatch(loader, data, labels):
      count += 1
    debug &"loaded {count} batches"
    check count == 100
    loader.shutdown()