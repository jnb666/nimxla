## The data module provides functions for loading common datasets and iterating over batches of data.

import std/[os, math, httpclient, strformat, strutils, sequtils, sugar, random, streams, endians, logging]
import zippy
import zippy/tarballs
import ../nimxla
import private/utils

type
  Dataset* = object
    ## Dataset represents a set of data samples from 0..len-1 where getItem(i) returns
    ## a tuple with the ith sample and it's classification label. Shape gives the size
    ## of each item - e.g. [rows, cols, 3] for a 2d RGB image.
    ## classes returns the classification labels.
    ## normalize returns the mean and standard deviation for each channel.
    name*: string
    getItem*: proc(i: int, data: pointer): int32 {.closure}
    shape*: seq[int]
    len*: int
    classes*: seq[string]
    normalize*: (seq[float32], seq[float32])

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

proc shape*(d: DataLoader): seq[int] =
  ## Shape of one batch of images returned from the dataset.
  @[d.batchSize] & d.dataset.shape

proc `$`*(d: Dataset): string =
  d.name & "[" & map(@[d.len] & d.shape, x => $x).join(" ") & "]"

proc cacheDir(subdir: string): string =
  ## Return location of cache directory + provided path - creates it if it does not exist
  let dir = joinPath(getCacheDir("nimxla"), subdir)
  debug &"cache dir is {dir}"
  if not dirExists(dir):
    createDir(dir)
  return dir  

proc download(baseurl: string, filenames: openarray[string], unzip = false) =
  ## Download given files from URL starting with baseurl and optionally unzip them.
  let client = newHttpClient()
  client.headers = newHttpHeaders({"Accept-Encoding": "gzip"})
  for file in filenames:
    if unzip:
      info &"downloading and unzipping {baseurl}{file}.gz"
      let resp = client.request(baseUrl & file & ".gz")
      let f = open(file, fmWrite)
      f.write(uncompress(resp.body))
      f.close
    else:
      info &"downloading {baseurl}{file}"
      client.downloadFile(baseUrl & file, file)

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
  (data, @[rows.int, cols.int, 1])

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
  ## Returned shape of each image is [28, 28, 1]
  const baseURL = "http://yann.lecun.com/exdb/mnist/"
  let origDir = getCurrentDir()
  setCurrentDir(cacheDir("mnist"))
  defer: setCurrentDir(origDir)
  let prefix = if train: "train" else: "t10k"
  let imageFile = prefix & "-images-idx3-ubyte"
  let labelFile = prefix & "-labels-idx1-ubyte"
  if not fileExists(imageFile) or not fileExists(labelFile):
    download(baseURL, [imageFile, labelFile], unzip=true)
  var (data, shape) = mnistImages(imageFile)
  let labels = mnistLabels(labelFile)
  let size = prod(shape)
  assert data.len == labels.len * size
  let classes = map(toSeq(0..9), x => $x)
  let normalization = (@[0.1307f32], @[0.3081f32])
  Dataset(
    name: &"MNIST(train={train})",
    getItem: proc(i: int, dout: pointer): int32 = 
      copyMem(dout, addr data[i*size], size)
      return labels[i].int32,
    shape: shape,
    len: labels.len,
    classes: classes,
    normalize: normalization
  )

proc getCifar10Batch(file: string, channels, chanSize: int, imgs, labels: var seq[uint8]) =
  debug &"read data from {file} imgSize={channels}*{chanSize}"
  let fs = openFileStream(file, fmRead)
  const batchSize = 10000
  var buf = newSeq[uint8](channels * chanSize)
  for i in 1 .. batchSize:
    labels.add fs.readUint8()
    let n = fs.readData(buf[0].addr, buf.len)
    assert n == buf.len
    # convert each image from C,H,W to H,W,C layout
    for j in 0 ..< chanSize:
      imgs.add buf[j]
      imgs.add buf[j+chanSize]
      imgs.add buf[j+2*chanSize]
  fs.close

proc cifar10Dataset*(train = false): Dataset =
  ## CIFAR10 dataset of 32x32 color images in 10 classes per http://www.cs.toronto.edu/~kriz/cifar.html
  const baseURL = "http://www.cs.toronto.edu/~kriz/"
  const tarFile = "cifar-10-binary.tar.gz"
  let shape   = @[32, 32, 3]
  let classes = @["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
  let origDir = getCurrentDir()
  setCurrentDir(cacheDir("cifar10"))
  defer: setCurrentDir(origDir)
  if not fileExists(tarFile):
    download(baseURL, [tarFile])
  if walkFiles("data/cifar-10-batches-bin/*.bin").toSeq.len < 6:
    debug "untar archive"
    removeDir("data")
    extractAll(tarFile, "data")
  let normalization = (@[0.4914f32, 0.4822, 0.4465], @[0.2023f32, 0.1994, 0.2010])
  let size = prod(shape)
  var data, labels: seq[uint8]
  if train:
    for i in 1..5:
      getCifar10Batch(&"data/cifar-10-batches-bin/data_batch_{i}.bin", 3, size div 3, data, labels)
  else:
    getCifar10Batch("data/cifar-10-batches-bin/test_batch.bin", 3, size div 3, data, labels)
  Dataset(
    name: &"CIFAR10(train={train})",
    getItem: proc(i: int, dout: pointer): int32 =
      copyMem(dout, addr data[i*size], size)
      return labels[i].int32,
    shape: shape,
    len: labels.len,
    classes: classes,
    normalize: normalization
  )