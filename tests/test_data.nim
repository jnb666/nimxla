{.warning[BareExcept]:off.}
import std/[unittest, logging, strutils]
import nimxla
import nimxla/data

const debug {.booldefine.} = false


suite "data":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)

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
    
  test "loader":
    setPrintOpts(minWidth = 4)
    let batch = 20
    let dataset = mnistDataset(train = true)
    let loader = initLoader(dataset, batchSize = batch)
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

