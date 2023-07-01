## The image module provides some simple functions for image augmentation. 
##
## Image arrays are stored as a Tensor[uint8] with dimensions [N,H,W,C] where N=number of images, HxW is image size and 
## C is number of channels (1 for greyscale or 3 for RGB).

{.warning[BareExcept]:off.}
import std/[math, random, strformat, strutils, sequtils, sugar, logging, os, cpuinfo]
import private/utils
import tensor

type
  Direction* = enum Horizontal, Vertical

  ImageOpKind* = enum Affine, Flip, Wrap, Elastic

  ImageOp* = object
    ## Apply operation to image with probability p
    p*: float
    case kind: ImageOpKind
    of Flip:
      ## Flip image around horizontal or vertical axis
      direction*: Direction
    of Wrap:
      ## Randomly scroll image in x and y direction by up to max_x, max_y wrapping pixels. 
      max_x*, max_y*: int
    of Affine:
      ## Affine image transform with rotation, scaling and translation relative to center
      rotate*:  float
      scalexy*: (float, float)
      transx*:  float
      transy*:  float
    of Elastic:
      ## Elastic image distortion
      kernelSize*: int
      stddev*: float
      scale*:  float
      kernel:  seq[float32]

  Image[C: static int] = object
    ## Image with 8 bits of data per channel. C is the number of channels - e.g. 1 for grayscale or 3 for RGB
    height:  int
    width:   int
    pixels:  seq[uint8]

  ImageRequest* = object
    height*: int
    width*:  int
    data*:   ptr uint8

  TransContext* = object
    cin*:  ptr Channel[ImageRequest]
    cout*: ptr Channel[bool]
    seed:  int64
    ops:   ptr UncheckedArray[ImageOp]
    nops:  int

  Transformer* = object
    ops*:    seq[ImageOp]
    ctx*:    TransContext
    threads: seq[Thread[TransContext]]


proc newImage[C: static int](width, height: int): Image[C] =
  ## Allocate a new blank image
  Image[C](width: width, height: height, pixels: newSeq[uint8](C*width*height))

proc len[C: static int](m: Image[C]): int {.inline.} =
  C * m.width * m.height

proc `[]`[C: static int](m: Image[C], x, y: int): array[C, uint8] {.inline.} =
  ## Get pixel from image at position (x, y). Will return zeros if x or y are out of range.
  if x >= 0 and x < m.width and y >= 0 and y < m.height:
    let pos = C * (y*m.width + x)
    copyMem(result.addr, unsafeAddr m.pixels[pos], C)

proc `[]=`[C: static int](m: var Image[C], x, y: int, pix: array[C, uint8]) {.inline.} =
  ## Set pixel at position (x, y) to val. No-op if x or y are out of range.
  if x >= 0 and x < m.width and y >= 0 and y < m.height:
    let pos = C * (y*m.width + x)
    copyMem(addr m.pixels[pos], pix.unsafeAddr, C)


proc gaussian(sigma: float32, size: int): seq[float32] =
  result = newSeq[float32](2*size+1)
  for x in -size .. size:
    result[x+size] = exp(-(x.float32*x.float32)/(2*sigma*sigma)) / (sqrt(2*PI) * sigma)

proc randomFlip*(direction: Direction, p = 0.5): ImageOp =
  ## Randomly flip images either horizontally or vertically with probability p.
  ImageOp(kind: Flip, p: p, direction: direction)

proc randomWrap*(max_x, max_y: int, p = 0.5): ImageOp =
  ## Translate and wrap around images such that the point which was at (0, 0) is at (dx, dy) 
  ## where dx is randomly sampled from -max_x..max_x and dy from -max_y..max_y. 
  ## p is probability of applying the transform to each image.
  ImageOp(kind: Wrap, p: p, max_x: max_x, max_y: maxy)

proc randomAffine*(rotate = 0.0, scale = (1.0, 1.0), transx = 0.0, transy = 0.0, p = 0.5): ImageOp =
  ## Random affine transformation with probability p, where image is rotated by up to -rotate..+rotate degrees, 
  ## scaled by factor between scale[0]..scale[1] and translated by up to transx and transy pixels along x and y. 
  ## These transforms are all relative to the image center.
  ImageOp(kind: Affine, p: p, rotate: rotate, scalexy: scale, transx: transx, transy: transy)

proc randomElastic*(kernelSize = 9, stddev = 4.0, scale = 0.5, p = 0.5): ImageOp =
  ## Random elastic transform where kernelSize is the size of the gaussian kernel, stddev is the x and y standard deviation for the kernel,
  ## scale is the scaling factor controlling the intensity of the deformation and p is the probabilty of applying the transform.
  ImageOp(kind: Elastic, p: p, kernelSize: kernelSize, stddev: stddev, scale: scale, kernel: gaussian(stddev, kernelSize))

proc `$`*(op: ImageOp): string =
  case op.kind
  of Flip:
    &"flip({op.direction} p={op.p})"
  of Wrap:
    &"wrap(x={op.max_x} y={op.max_y} p={op.p})"
  of Affine:
    var tr: seq[string]
    if op.rotate != 0: tr.add &"rotate={op.rotate}"
    if op.scalexy != (1.0, 1.0): tr.add &"scale={op.scalexy}"
    if op.transx != 0 or op.transy != 0: tr.add &"translate={op.transx},{op.transy}"
    "affine(" & tr.join(" ") & &" p={op.p})"
  of Elastic:
    &"elastic(kernel={op.kernelSize} stddev={op.stddev} scale={op.scale} p={op.p})"

proc `$`*(t: Transformer): string =
  map(t.ops, x => $x).join(" ")


proc flip[C: static int](m: Image[C], direction: Direction): Image[C] =
  result = newImage[C](m.width, m.height)
  let (mx, my) = (m.width-1, m.height-1)
  if direction == Horizontal:
    for y in 0 ..< m.height:
      for x in 0 ..< m.width:
        result[x, y] = m[mx-x, y]
  else:
    for y in 0 ..< m.height:
      for x in 0 ..< m.width:
        result[x, y] = m[x, my-y]

proc wrap[C: static int](m: Image[C], dx, dy: int): Image[C] =
  result = newImage[C](m.width, m.height)
  # scroll pixels
  let offset = C * (dy*m.width + dx)
  if offset > 0:
    copyMem(addr result.pixels[offset], unsafeAddr m.pixels[0], m.len-offset)
  else:
    copyMem(addr result.pixels[0], unsafeAddr m.pixels[-offset], m.len+offset)
  # fill edges with wrapped values
  let wc = C * m.width
  var (y1, y2) = (0, m.height)
  if dy > 0: # top
    y1 = dy
    copyMem(addr result.pixels[0], unsafeAddr m.pixels[m.len-wc*dy], wc*dy)
  if dy < 0: # bottom
    y2 = m.height + dy
    copyMem(addr result.pixels[m.len+wc*dy], unsafeAddr m.pixels[0], -wc*dy)
  if dx > 0: # left
    let sx = C * (m.width-dx)
    for y in y1 ..< y2:
      copyMem(addr result.pixels[y*wc], unsafeAddr m.pixels[y*wc+sx], C*dx)
  if dx < 0: # right
    let sx = C * (m.width+dx)
    for y in y1 ..< y2:
       copyMem(addr result.pixels[y*wc+sx], unsafeAddr m.pixels[y*wc], -C*dx)

proc interpolate[C: static int](m: Image[C], x, y: float32): array[C, uint8] =
  ## bilinear interpolation
  let (ix, iy) = (x.int, y.int)
  let p1 = m[ix, iy]
  let p2 = m[ix+1, iy]
  let p3 = m[ix, iy+1]
  let p4 = m[ix+1, iy+1]
  let (xf, yf) = (x-x.floor, y-y.floor)
  for i in 0 ..< C:
    let avg0 = p1[i].float32*(1-xf) + p2[i].float32*xf
    let avg1 = p3[i].float32*(1-xf) + p4[i].float32*xf 
    result[i] = uint8(clamp(avg0*(1-yf) + avg1*yf + 0.5, 0, 255))

proc affine[C: static int](m: Image[C], angle, sx, sy, tx, ty: float32): Image[C] =
  result = newImage[C](m.width, m.height)
  let (sina, cosa) = (sin(angle), cos(angle))
  let (xc, yc) = ((m.width.float32-1)/2, (m.height.float32-1)/2)
  let mat = [
    sx*cosa, -sx*sina, xc+tx, 
    sy*sina,  sy*cosa, yc+ty,
  ]
  for iy in 0 ..< m.height:
    let yp = iy.float32 - yc
    for ix in 0 ..< m.width:
      let xp = ix.float32 - xc
      let x = mat[0]*xp + mat[1]*yp + mat[2]
      let y = mat[3]*xp + mat[4]*yp + mat[5]
      result[ix, iy] = m.interpolate(x, y)

proc convH(dst: var seq[float32], src, kernel: seq[float32], ksize, w, h: int) =
  for x in 0 ..< w:
    let (x1, x2) = (max(x-ksize, 0), min(x+ksize, w-1))
    var sum: float32
    for ix in x1 .. x2:
      sum += kernel[x-ix+ksize]
    for y in 0 ..< h:
      var val: float32
      for ix in x1 .. x2:      
        val += src[ix+y*w] * kernel[x-ix+ksize]
      dst[x+y*w] = val / sum

proc convV(dst: var seq[float32], src, kernel: seq[float32], ksize, w, h: int) =
  for y in 0 ..< h:
    let (y1, y2) = (max(y-ksize, 0), min(y+ksize, h-1))
    var sum: float32
    for iy in y1 .. y2:
      sum += kernel[y-iy+ksize]
    for x in 0 ..< w:
      var val: float32
      for iy in y1 .. y2:      
        val += src[x+iy*w] * kernel[y-iy+ksize]
      dst[x+y*w] = val / sum

proc randomConvolve(rng: var Rand, kernel: seq[float32], ksize, w, h: int): seq[float32] =
  var tmp = newSeq[float32](w*h)
  result = newSeqWith(w*h, rng.rand(-1f32 .. 1f32))
  convH(tmp, result, kernel, ksize, w, h)
  convV(result, tmp, kernel, ksize, w, h)

proc elastic[C: static int](m: Image[C], dx, dy: seq[float32], scale: float32): Image[C] =
  result = newImage[C](m.width, m.height)
  let xscale = scale * m.width.float32
  let yscale = scale * m.height.float32
  for iy in 0 ..< m.height:
    let yp = iy.float32 + 0.5
    for ix in 0 ..< m.width:
      let xp = ix.float32 + 0.5
      let pos = iy*m.width + ix
      result[ix, iy] = m.interpolate(xp + dx[pos]*xscale, yp + dy[pos]*yscale)

proc transformImage[C: static int](img: Image[C], rng: var Rand, ops: ptr UncheckedArray[ImageOp], nops: int): Image[C] =
  var m = img
  for i in 0 ..< nops:
    let op = ops[i]
    if rng.rand(1.0) <= op.p:
      case op.kind:
      of Flip:
        m = flip[C](m, op.direction)
      of Wrap:
        let dx = rng.rand(-op.max_x .. op.max_x)
        let dy = rng.rand(-op.max_y .. op.max_y)
        m = wrap[C](m, dx, dy)
      of Affine:
        var angle: float32
        if op.rotate != 0:
          angle = rng.rand(-op.rotate .. op.rotate).degToRad
        var (sx, sy) = (1f32, 1f32)
        if op.scalexy != (1.0, 1.0):
          sx = rng.rand(op.scalexy[0] .. op.scalexy[1])
          sy = rng.rand(op.scalexy[0] .. op.scalexy[1])
        var tx, ty: float32
        if op.transx != 0.0:
          tx = rng.rand(op.transx)
        if op.transy != 0.0:
          ty = rng.rand(op.transy)
        m = affine[C](m, angle, sx, sy, tx, ty)
      of Elastic:
        let dx = randomConvolve(rng, op.kernel, op.kernelSize, m.width, m.height)
        let dy = randomConvolve(rng, op.kernel, op.kernelSize, m.width, m.height)
        m = elastic[C](m, dx, dy, op.scale)
  return m

proc imageTransformThread[C: static int](ctx: TransContext) {.thread.} =
  var rng = initRand(ctx.seed)
  let cin = ctx.cin
  let cout = ctx.cout
  while true:
    let req = cin[].recv
    if req.width == 0:
      return
    let size = C * req.width * req.height
    var img = newImage[C](req.width, req.height)
    copyMem(img.pixels[0].addr, req.data, size)
    img = transformImage(img, rng, ctx.ops, ctx.nops)
    copyMem(req.data, img.pixels[0].addr, size)
    cout[].send(true)

proc initTransformer*(channels: static int, rng: var Rand, ops: openarray[ImageOp], threads = 0): Transformer =
  ## Setup a new image transform pipeline which will read apply a series of image ops.
  ## Will use threads parallel threads - sets this to no. of processor cores if <= 0
  if ops.len == 0:
    return
  result.ops = @ops
  let nthreads = if threads <= 0: countProcessors() else: threads
  result.threads = newSeq[Thread[TransContext]](nthreads)
  result.ctx.cin = newChannel[ImageRequest]()
  result.ctx.cout = newChannel[bool]()
  result.ctx.ops = cast[ptr UncheckedArray[ImageOp]](ops[0].unsafeAddr)
  result.ctx.nops = ops.len
  debug &"init transformer with {nthreads} threads: {ops}"
  for i in 0 .. high(result.threads):
    result.ctx.seed = rng.rand(high(int64))
    createThread(result.threads[i], imageTransformThread[channels], result.ctx)

proc transform*(t: Transformer, arr: var Tensor[uint8]) =
  ## Apply series of image transforms to an array of images in [N,H,W,C] format.
  ## The transform will be applied in parallel using multiple threads - the 
  ## proc will return once all are completed and the data is updated.
  if arr.dims.len != 4:
    raise newException(ValueError, &"getImage: expecting 4d input tensor - got {arr.dims}")
  if t.ops.len == 0:
    return
  for i in 0 ..< arr.dims[0]:
    let img = arr.at(i)
    t.ctx.cin[].send ImageRequest(height: img.dims[0], width: img.dims[1], data: img.rawPtr)
  for i in 0 ..< arr.dims[0]:
    discard t.ctx.cout[].recv

proc shutdown*(t: var Transformer) =
  ## Shutdown worker threads
  if t.ops.len > 0:
    debug "shutdown transformer"
    var done: ImageRequest
    for i in 0 .. high(t.threads):
      t.ctx.cin[].send done
    sleep(100)
    t.ctx.cin[].close
    while true:
      let (ok, msg) = t.ctx.cout[].tryRecv
      if not ok: break
    t.ctx.cout[].close
    t.ops.setLen(0)
