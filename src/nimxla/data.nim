## The data module provides functions for loading common datasets and iterating over batches of data.

import std/[os, math, httpclient, strformat, sequtils, random, streams, endians, logging]
import zippy
import ../nimxla
import private/utils

type
  Dataset* = object
    ## Dataset represents a set of data samples from 0..length-1 where getItem(i) returns
    ## a tuple with the ith sample and it's classification label. Shape gives the size
    ## of each item - e.g. [rows, cols] for a 2d image.
    name*: string
    getItem*: proc(i: int, data: pointer): int32 {.closure}
    shape*: seq[int]
    len*: int

  DataLoader* = object
    ## DataLoader provides an iterator to read batches of data from a dataset.
    dataset*:   Dataset
    batchSize*: int
    shuffle*:   bool


proc initLoader*(d: Dataset, batchSize = 0, shuffle = false): DataLoader =
  ## Create a new loader for the given dataset. If batch size is 0 then
  ## the batch size defaults to the dataset length.
  result.dataset = d
  result.batchSize = if batchSize == 0: d.len else: batchSize
  result.shuffle = shuffle

iterator getBatch*[T](d: DataLoader, data: var Tensor[T], labels: var Tensor[int32]): int {.closure.} =
  ## Returns batches of data from the dataset associated with the loader.
  ## Output data and labels should be preallocated with size of one batch of data.
  assert data.len == d.batchSize * prod(d.dataset.shape)
  assert labels.len == d.batchSize
  var indexes = toSeq(0 ..< d.dataset.len)
  if d.shuffle: shuffle(indexes)
  let size = prod(d.dataset.shape)
  for batch in 0 ..< d.dataset.len div d.batchSize:
    for i in 0 ..< d.batchSize:
      let ix = indexes[batch*d.batchSize + i]
      labels[i] = d.dataset.getItem(ix, ptrOffset(data.rawPtr, i*size))
    yield batch

proc `$`*(d: Dataset): string =
  d.name

proc cacheDir(): string =
  ## Return location of cache directory - creates it if it does not exist
  let dir = os.getCacheDir("nimxla")
  if not dirExists(dir):
    createDir(dir)
  debug &"cache dir is {dir}"
  return dir  

proc download(baseurl: string, filenames: varargs[string]) =
  ## Download and uncompress given files.
  let client = newHttpClient()
  client.headers = newHttpHeaders({"Accept-Encoding": "gzip"})
  for file in filenames:
    info &"downloading {baseurl}{file}"
    let resp = client.request(baseUrl & file & ".gz")
    let f = open(file, fmWrite)
    f.write(uncompress(resp.body))
    f.close

proc readInt32BE(stream: Stream): int32 =
  var bytes = stream.readInt32
  bigEndian32(addr result, addr bytes)

proc mnistImages(file: string): (seq[uint8], seq[int]) =
  let fs = openFileStream(file, fmRead)
  let magic = fs.readInt32BE
  assert magic == 0x0803
  let num = fs.readInt32BE
  let rows = fs.readInt32BE
  let cols = fs.readInt32BE
  let bytes = num*rows*cols
  var data = newSeq[uint8](bytes)
  let n = fs.readData(addr data[0], bytes)
  assert n == bytes
  debug &"got {num} {rows}x{cols} mnist images"
  fs.close
  (data, @[rows.int, cols.int])

proc mnistLabels(file: string): seq[uint8] =
  let fs = openFileStream(file, fmRead)
  let magic = fs.readInt32BE
  assert magic == 0x0801
  let num = fs.readInt32BE
  result = newSeq[uint8](num)
  let n = fs.readData(addr result[0], num)
  assert n == num
  debug &"got {num} mnist labels"
  fs.close 

proc mnistDataset*(train = false): DataSet =
  ## MNIST dataset of handwritten digits per http://yann.lecun.com/exdb/mnist/
  ## will download the data to and save a cached copy.
  const baseURL = "http://yann.lecun.com/exdb/mnist/"
  setCurrentDir(cacheDir())
  let prefix = if train: "train" else: "t10k"
  let imageFile = prefix & "-images-idx3-ubyte"
  let labelFile = prefix & "-labels-idx1-ubyte"
  if not fileExists(imageFile) or not fileExists(labelFile):
    download(baseURL, imageFile, labelFile)
  var (data, shape) = mnistImages(imageFile)
  let labels = mnistLabels(labelFile)
  let size = prod(shape)
  assert data.len == labels.len * size
  Dataset(
    name: &"MNIST(train={train})",
    getItem: proc(i: int, dout: pointer): int32 = 
      copyMem(dout, addr data[i*size], size)
      return labels[i].int32,
    shape: shape,
    len: labels.len
  )


