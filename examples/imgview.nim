# simple viewer for MNIST test and training images 
# params to the main proc can be set via cmd line argmuments
# labels parameter takes the output file from the mnist_mlp or mnist_cnn program
# and filters the plots to only show the errors

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

proc filterErrors(dset: Dataset, indexes: seq[int], file: string): seq[int] = 
  let pred = readTensor[int32](file)
  echo "read predictions: ", pred.shape
  if pred.len != dset.len:
    quit("Error: prediction file size does not match dataset")
  echo "filtering to return only errors"
  var t = newTensor[uint8](dset.shape)
  for ix in indexes:
    let label = dset.getItem(ix, t.rawPtr)
    if pred[ix] != label:
      result.add ix

proc main(page = 1, rows = 5, cols = 8, class = "", input = "") =
  let ws = openWebSocket()
  let dset = mnistDataset(train=false)
  echo dset
  var indexes = dset.filterClass(class)
  if input != "":
    indexes = filterErrors(dset, indexes, input)

  echo &"got {indexes.len} enties"
  var t = newTensor[uint8](dset.shape)
  let blank = zeros[uint8](dset.shape)

  proc getData(i: int): (Tensor[uint8], string) =
    let n = (page-1)*rows*cols + i
    if n < 0 or n >= indexes.len:
      return (blank, "")
    let label = dset.getItem(indexes[n], t.rawPtr)
    (t, &"{n}: {label}")

  let pages = (indexes.len-1) div (rows*cols) + 1
  let title = &"page {page} of {pages}"
  let (imgs, layout) = plotImageGrid(title, rows, cols, getData)

  updatePlot(ws, imgs, layout)
  ws.close


dispatch main
