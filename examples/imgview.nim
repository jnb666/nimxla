# simple viewer for MNIST or CIFAR10 test images 
# params to the main proc can be set via cmd line argmuments
# input parameter takes the output file from the mnist_mlp or mnist_cnn program
# errors flag filters the plots to only show the errors

{.warning[BareExcept]:off.}
import std/[strformat, sequtils, strformat, logging, random, times]
import nimxla
import nimxla/[data, plots, image]
import cligen
import ws

proc filterClass(dset: Dataset, class: string): seq[int] =
  if class == "":
    return toSeq(0 ..< dset.len)
  echo "filtering for label: ", class
  var t = newTensor[uint8](dset.shape)
  for i in 0 ..< dset.len:
    let label = dset.getItem(i, t.rawPtr)
    if dset.classes[label] == class:
      result.add i

proc filterErrors(dset: Dataset, indexes: seq[int], pred: Tensor[int32]): seq[int] = 
  echo "filtering to return only errors"
  var t = newTensor[uint8](dset.shape)
  for ix in indexes:
    let label = dset.getItem(ix, t.rawPtr)
    if pred[ix] != label:
      result.add ix

proc main(page = 1, rows = 8, cols = 12, class = "", input = "", errors = false, cifar10 = false,
          flip = false, wrap = 0, rotate = false, scale = false, elastic = 0.0, debug = false) =
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  var rng = initRand(getTime().toUnix * 1_000_000_000 + getTime().nanosecond)
  let ws = openWebSocket()

  var ops: seq[ImageOp]
  if flip:
    ops.add randomFlip(Horizontal)
  if wrap != 0:
    ops.add randomWrap(wrap, wrap)
  if rotate:
    ops.add randomAffine(rotate=15)
  if scale:
    ops.add randomAffine(scale=(0.85,1.15))
  if elastic != 0:
    ops.add randomElastic(scale=elastic)

  var dset: Dataset
  var trans: Transformer
  if cifar10:
    dset = cifar10Dataset()
    trans = initTransformer(3, rng, ops)
  else:
    dset = mnistDataset()
    trans = initTransformer(1, rng, ops)
  echo dset

  var indexes = dset.filterClass(class)
  var pred: Tensor[int32]
  if input != "":
    pred = readTensor[int32](input)
    echo "read predictions: ", pred.shape
    if pred.len != dset.len:
      quit("Error: prediction file size does not match dataset")
    if errors:
      indexes = filterErrors(dset, indexes, pred)
  echo &"got {indexes.len} enties"

  proc getData(): (Tensor[uint8], seq[string]) =
    var imgs = zeros[uint8](@[rows*cols] & dset.shape)
    var text = newSeq[string](rows*cols)
    for i in 0 ..< rows*cols:
      let n = (page-1)*rows*cols + i
      if n >= 0 and n < indexes.len:
        discard dset.getItem(indexes[n], imgs.at(i).rawPtr)
        text[i] = $indexes[n] & ":"
        if input != "":
          text[i] &= " " & dset.classes[pred[indexes[n]]]
    trans.transform(imgs)
    return (imgs, text)

  let pages = (indexes.len-1) div (rows*cols) + 1
  var title = &"page {page} of {pages}"
  if class != "": title.add &"  class={class}"
  if errors and input != "": title.add "  errors only"
  if ops.len > 0:
    title.add &"  transforms={trans}"
  let (imgs, layout) = plotImageGrid(title, rows, cols, getData)

  updatePlot(ws, imgs, layout)
  ws.close


dispatch main
