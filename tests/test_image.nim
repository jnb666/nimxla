{.warning[BareExcept]:off.}
import std/[unittest, random, logging]
import nimxla
import nimxla/[image, data]

const debug {.booldefine.} = false

suite "image":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  var rng = initRand(0)

  test "flip":
    setPrintOpts(minWidth = 4)
    let d = mnistDataset(train = false)
    var img = newTensor[uint8](1, 28, 28, 1)
    let label = d.getItem(0, img.rawPtr)
    debug img.reshape(28, 28)
    check img[0, 26, 10, 0] == 121
    check img[0, 26, 17, 0] == 0

    var trans = initTransformer(1, rng, [randomFlip(Horizontal, p=1.0)])
    trans.transform(img)
    debug img.reshape(28, 28)
    check img[0, 26, 10, 0] == 0
    check img[0, 26, 17, 0] == 121
    setPrintOpts()

    trans.shutdown()