# simple viewer for MNIST or CIFAR10 test images 
# params to the main proc can be set via cmd line argmuments
# input parameter takes the output file from the mnist_mlp or mnist_cnn program
# errors flag filters the plots to only show the errors

import std/[strformat, sequtils, strformat]
import nimxla
import nimxla/[data, plots]
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

proc main(page = 1, rows = 8, cols = 12, class = "", input = "", errors = false, cifar10 = false) =
  let ws = openWebSocket()
  var dset: Dataset
  if cifar10:
    dset = cifar10Dataset()
  else:
    dset = mnistDataset()
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
  var t = newTensor[uint8](dset.shape)
  let blank = zeros[uint8](dset.shape)

  proc getData(i: int): (Tensor[uint8], string) =
    let n = (page-1)*rows*cols + i
    if n < 0 or n >= indexes.len:
      return (blank, "")
    let label = dset.getItem(indexes[n], t.rawPtr)
    let text = if input != "":
      dset.classes[pred[indexes[n]]]
    else:
      ""
    (t, &"{n}: {text}")

  let pages = (indexes.len-1) div (rows*cols) + 1
  let title = &"page {page} of {pages}"
  let (imgs, layout) = plotImageGrid(title, rows, cols, getData)

  updatePlot(ws, imgs, layout)
  ws.close


dispatch main
