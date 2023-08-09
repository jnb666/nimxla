## The data module provides functions for loading common datasets and iterating over batches of data.

{.warning[BareExcept]:off.}
import std/[os, math, httpclient, strformat, strutils, sequtils, sugar, random, streams, endians, macros, logging, atomics]
import zippy
import zippy/tarballs
import ../nimxla
import image
import private/utils

type
  Dataset* = ref object of RootRef
    ## Dataset base class.
    name:    string
    labels:  seq[int32]
    shape:   seq[int]
    classes: seq[string]
    mean:    seq[float32]
    stddev:  seq[float32]

  MNISTDataset* = ref object of Dataset
    data:   seq[uint8]

  CIFAR10Dataset* = ref object of Dataset
    data:   seq[uint8]

  LoaderContext = object
    cout:      ptr Channel[(int, ptr uint8, ptr int32)]
    tin:       ptr Channel[ImageRequest]
    tout:      ptr Channel[bool]
    seed:      int64
    batchSize: int
    shuffle:   bool
    workers:   ptr Atomic[int]
    shutdown:  ptr Atomic[bool]

  DataLoader* = object
    ## DataLoader provides an iterator to read batches of data from a dataset.
    ## C is the number of image channels
    dataset*:   Dataset
    batchSize*: int
    shuffle*:   bool
    rng:        Rand
    trans:      Transformer
    ctx:        LoaderContext
    thread:     Thread[LoaderContext]


method getItem*(d: Dataset, i: int, data: pointer): int32 {.base, gcsafe.} =
  ## Get ith entry and copy data to to data Tensor. Returns label.
  raise newException(ValueError, "abstract method")

proc len*(d: Dataset): int =
  ## Number of items in dataset
  d.labels.len

proc shape*(d: Dataset): seq[int] =
  ## Get shape of one element of dataset.
  d.shape

proc classes*(d: Dataset): seq[string] =
  ## Get list of class names.
  d.classes

proc normalization*(d: Dataset): (seq[float32], seq[float32]) =
  ## Get per channel mean and std deviation for normalization
  (d.mean, d.stddev)

proc `$`*(d: Dataset): string =
  ## Dataset name
  d.name & "[" & map(@[d.len] & d.shape, x => $x).join(" ") & "]"

proc cacheDir(subdir: string): string =
  ## Return location of cache directory + provided path - creates it if it does not exist
  let dir = joinPath(getCacheDir("nimxla"), subdir)
  debug &"cache dir is {dir}"
  if not dirExists(dir):
    createDir(dir)
  return dir

proc download(baseurl, dir: string, filenames: openarray[string], unzip = false) =
  ## Download given files from URL starting with baseurl and optionally unzip them.
  let client = newHttpClient()
  client.headers = newHttpHeaders({"Accept-Encoding": "gzip"})
  for file in filenames:
    if unzip:
      info &"downloading and unzipping {baseurl}{file}.gz"
      let resp = client.request(baseUrl & file & ".gz")
      let f = open(joinPath(dir, file), fmWrite)
      f.write(uncompress(resp.body))
      f.close
    else:
      info &"downloading {baseurl}{file}"
      client.downloadFile(baseUrl & file, joinPath(dir, file))

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

proc mnistLabels(file: string): seq[int32] =
  let fs = openFileStream(file, fmRead)
  let magic = fs.readInt32BE
  assert magic == 0x0801
  let num = fs.readInt32BE
  var labels = newSeq[uint8](num)
  let n = fs.readData(addr labels[0], num)
  assert n == num
  debug &"got {num} mnist labels"
  fs.close
  map(labels, x => x.int32)

proc mnistDataset*(train = false): MNISTDataset =
  ## MNIST dataset of handwritten digits per http://yann.lecun.com/exdb/mnist/
  ## will download the data to and save a cached copy.
  ## Returned shape of each image is \[28, 28, 1\]
  const baseURL = "http://yann.lecun.com/exdb/mnist/"
  let dir = cacheDir("mnist")
  let prefix = if train: "train" else: "t10k"
  let imageFile = prefix & "-images-idx3-ubyte"
  let labelFile = prefix & "-labels-idx1-ubyte"
  if not fileExists(joinPath(dir, imageFile)) or not fileExists(joinPath(dir, labelFile)):
    download(baseURL, dir, [imageFile, labelFile], unzip=true)
  result = MNISTDataset(
    name:    &"MNIST(train={train})",
    classes: map(toSeq(0..9), x => $x),
    mean:    @[0.1307f32],
    stddev:  @[0.3081f32],
  )
  (result.data, result.shape) = mnistImages(joinPath(dir, imageFile))
  result.labels = mnistLabels(joinPath(dir, labelFile))
  assert result.data.len == result.labels.len * prod(result.shape)

method getItem*(d: MNISTDataset, i: int, dout: pointer): int32 =
  ## Get ith entry and copy data to to data Tensor. Returns label.
  let size = prod(d.shape)
  copyMem(dout, addr d.data[i*size], size)
  return d.labels[i].int32


proc getCifar10Batch(file: string, imgs: var seq[uint8], labels: var seq[int32]) =
  const channels = 3
  const chanSize = 32*32
  debug &"read data from {file} imgSize={channels}*{chanSize}"
  let fs = openFileStream(file, fmRead)
  const batchSize = 10000
  var buf = newSeq[uint8](channels * chanSize)
  for i in 1 .. batchSize:
    labels.add fs.readUint8().int32
    let n = fs.readData(buf[0].addr, buf.len)
    assert n == buf.len
    # convert each image from C,H,W to H,W,C layout
    for j in 0 ..< chanSize:
      imgs.add buf[j]
      imgs.add buf[j+chanSize]
      imgs.add buf[j+2*chanSize]
  fs.close

proc cifar10Dataset*(train = false): CIFAR10Dataset =
  ## CIFAR10 dataset of 32x32 color images in 10 classes per http://www.cs.toronto.edu/~kriz/cifar.html
  const baseURL = "http://www.cs.toronto.edu/~kriz/"
  const tarFile = "cifar-10-binary.tar.gz"
  let dir = cacheDir("cifar10")
  if not fileExists(joinPath(dir, tarFile)):
    download(baseURL, dir, [tarFile])
  let dataDir = joinPath(dir, "data")
  if walkFiles(dataDir & "/cifar-10-batches-bin/*.bin").toSeq.len < 6:
    debug "untar archive"
    removeDir(dataDir)
    extractAll(joinPath(dir, tarFile), dataDir)
  result = CIFAR10Dataset(
    name:    &"CIFAR10(train={train})",
    classes: @["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    shape:   @[32, 32, 3],
    mean:    @[0.4914f32, 0.4822, 0.4465],
    stddev:  @[0.2023f32, 0.1994, 0.2010],
  )
  if train:
    for i in 1..5:
      getCifar10Batch(dataDir & &"/cifar-10-batches-bin/data_batch_{i}.bin", result.data, result.labels)
  else:
    getCifar10Batch(dataDir & &"/cifar-10-batches-bin/test_batch.bin", result.data, result.labels)

method getItem*(d: CIFAR10Dataset, i: int, dout: pointer): int32 =
  ## Get ith entry and copy data to to data Tensor. Returns label.
  let size = prod(d.shape)
  copyMem(dout, addr d.data[i*size], size)
  return d.labels[i].int32


proc newLoader*(rng: var Rand, batchSize = 0, shuffle = false): DataLoader =
  ## Create a new loader for the given dataset. If batch size is 0 then the batch size defaults to the dataset length.
  result.rng = rng
  result.batchSize = batchSize
  result.shuffle = shuffle

proc shape*(d: DataLoader): seq[int] =
  ## Shape of one batch of images returned from the dataset.
  @[d.batchSize] & d.dataset.shape

iterator getBatch*[T](d: var DataLoader, data: var Tensor[T], labels: var Tensor[int32]): int {.closure.} =
  ## Returns batches of data from the dataset associated with the loader.
  ## Output data and labels should be preallocated with size of one batch of data.
  assert data.len == d.batchSize * prod(d.dataset.shape)
  assert labels.len == d.batchSize
  if d.ctx.cout == nil:
    var indexes = toSeq(0 ..< d.dataset.len)
    if d.shuffle: d.rng.shuffle(indexes)
    for batch in 0 ..< d.dataset.len div d.batchSize:
      for i in 0 ..< d.batchSize:
        let ix = indexes[batch*d.batchSize + i]
        labels[i] = d.dataset.getItem(ix, data.at(i).rawPtr)
      if d.trans.ops.len > 0:
        d.trans.transform(data)
      yield batch
  else:
    while true:
      let (batch, dptr, lptr) = d.ctx.cout[].recv
      if batch < 0:
        break
      copyMem(data.rawPtr, dptr, data.len)
      copyMem(labels.rawPtr, lptr, 4*labels.len)
      yield batch

proc `$`*(d: DataLoader): string =
  result = &"{d.dataset} batchSize={d.batchSize} shuffle={d.shuffle}"
  if d.trans.ops.len > 0:
    result.add &"\n transform=[{d.trans}]"

proc loaderThread(ctx: LoaderContext, dataset: Dataset) =
  var rng = initRand(ctx.seed)
  var indexes = toSeq(0 ..< dataset.len)
  let batchSize = ctx.batchSize
  let dims = @[batchSize] & dataset.shape
  var bufix = 0
  var buffer: array[3, Tensor[uint8]]
  var labels: array[3, Tensor[int32]]
  for i in 0..2:
    buffer[i] = newTensor[uint8](dims)
    labels[i] = newTensor[int32](batchSize)
  while true:
    if ctx.shuffle: rng.shuffle(indexes)
    for batch in 0 ..< dataset.len div batchSize:
      # poll for shutdown signal
      if load(ctx.shutdown[]):
        atomicDec(ctx.workers[])
        return
      # transform batch of images
      for i in 0 ..< batchSize:
        let ix = indexes[batch*batchSize + i]
        let t = buffer[bufix].at(i)
        labels[bufix][i] = dataset.getItem(ix, t.rawPtr)
        ctx.tin[].send ImageRequest(height: t.dims[0], width: t.dims[1], data: t.rawPtr)
      for i in 0 ..< batchSize:
        discard ctx.tout[].recv
      # send response
      ctx.cout[].send (batch, buffer[bufix].rawPtr, labels[bufix].rawPtr)
      bufix = (bufix + 1) mod 3
    # signal end of epoch
    ctx.cout[].send (-1, nil, nil)

template start*(d: var DataLoader, dset: untyped, channels: static int, transforms: varargs[ImageOp]) =
  ## Start should be called to associate a DatatSet with the loader.
  ## If the optional image transforms are set then it will start a new thread to launch the 
  ## image augmentation process in the background. In this case channels should be set to the number
  ## of color channels.
  `d`.dataset = `dset`
  if `d`.batchSize == 0:
    `d`.batchSize = `d`.dataset.len
  `d`.trans = initTransformer(`channels`, `d`.rng, `transforms`)
  if `transforms`.len > 0:
    proc spawnThread(ctx: LoaderContext) {.thread.} =
      loaderThread(ctx, `dset`)
    debug "spawn loader thread"
    `d`.ctx.cout = newChannel[(int, ptr uint8, ptr int32)](maxItems = 1)
    `d`.ctx.tin = `d`.trans.ctx.cin
    `d`.ctx.tout = `d`.trans.ctx.cout
    `d`.ctx.seed = `d`.rng.rand(high(int64))
    `d`.ctx.shuffle = `d`.shuffle
    `d`.ctx.batchSize = `d`.batchSize
    `d`.ctx.shutdown = newAtomic(false)
    `d`.ctx.workers = newAtomic(1)
    createThread(`d`.thread, spawnThread, `d`.ctx)

proc shutdown*(d: DataLoader) =
  ## Shutdown worker thread
  if d.ctx.workers != nil:
    debug "shutdown DataLoader..."
    store(d.ctx.shutdown[], true)
    while true:
      let (ok, msg) = d.ctx.cout[].tryRecv
      if not ok: break
    while load(d.ctx.workers[]) > 0:
      sleep(20)
    debug "..done"
    freeChannel(d.ctx.cout)
    deallocShared(d.ctx.workers)
    deallocShared(d.ctx.shutdown)
  d.trans.shutdown