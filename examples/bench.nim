# run a matrix multiply and add calculation repeatedly to get a rough idea of the performance
# [depends on https://github.com/c-blake/cligen for arg parsing]
# 
# some example stats with Cuda on a Nvidia RTX 3080:
#
#  $ nim -d:release r bench.nim  -d=10   --gpu --dtype=F32
#  gpu version:cuda 11080 devices:1
#  matrix size: 1024 x 1024 data type: F32
#  running benchmark for 10.0 seconds
#  processed 111395 calls - 89.8 µs/call
#  
#  $ nim -d:release r bench.nim  -d=10   --gpu --dtype=F16
#  gpu version:cuda 11080 devices:1
#  matrix size: 1024 x 1024 data type: F16
#  running benchmark for 10.0 seconds
#  processed 208363 calls - 48.0 µs/call

import std/[sequtils, math, random, times, strformat, strutils]
import nimxla
import cligen

template timeIt(duration: float, code: untyped): untyped =
  echo &"running benchmark for {duration} seconds"
  let start = epochTime()
  var ncalls = 0 
  while epochTime()-start < duration:
    `code`
    ncalls += 1
  let elapsed = epochTime() - start
  let msPerCall = 1e6*elapsed/float(ncalls)
  echo "processed $1 calls - $2 µs/call" % [$ncalls, msPerCall.formatFloat(ffDecimal, 1)]

proc buildExec(c: Client, n: int, typ: DataType, debug: bool): Executable =
  let b = newBuilder("benchmark")
  let x = b.parameter(typ, [n, n])
  let y = b.parameter(typ, [n, n])
  let z = b.parameter(typ, [1, n])
  let comp = b.build dot(x, y) + z
  if debug: echo comp.last.repr
  c.compile(comp)

proc randomTensor(typ: DataType, dims: openarray[int], minVal, maxVal: float32): Literal =
  let vals = newSeqWith(prod(dims), rand(minVal..maxVal))
  let t = vals.toTensor.reshape(dims)
  t.toLiteral.convert(typ)

proc makeBuffers(c: Client, n:int, typ: DataType): seq[Buffer] =
  echo &"matrix size: {n} x {n} data type: {typ}"
  randomize()
  let x = randomTensor(typ, [n, n], 0, 1)
  let y = randomTensor(typ, [n, n], 0, 1)
  let z = randomTensor(typ, [1, n], -0.01, 0.01)
  @[c.newBuffer(x), c.newBuffer(y), c.newBuffer(z)]

proc main(gpu=false, size=1024, duration=1.0, dtype="F32", debug=false) =
  let c = if gpu: newGPUClient() else: newCPUClient()
  echo c
  let typ = parseEnum[DataType](dtype)
  let exec = c.buildExec(size, typ, debug)
  let buffers = c.makeBuffers(size, typ)
  timeIt(duration):
    discard exec.run(buffers)

dispatch main
