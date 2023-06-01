{.warning[BareExcept]:off.}
import std/[unittest, logging, strutils, sequtils, random]
import nimxla

const debug {.booldefine.} = false

suite "tensor":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)

  let t1 = toTensor[float32](1..24).reshape(2, 3, 4)

  test "scalar":
    let s = 42.0.toTensor
    debug s
    check s.len == 1
    check s.dims == []
    check s[] == 42.0
    check s is Tensor[float64]

  test "vector":
    var v = [2'i32, 3, 5, 7].toTensor
    debug v
    for ix in 0 ..< v.len:
      v[ix] = v[ix] * v[ix]
    debug v
    check v.len == 4
    check v.dims == [4]
    check v.toSeq == [4'i32, 9, 25, 49]

  test "matrix":
    let m = fill([2, 3], 0.5f32)
    debug m
    check m.len == 6
    check m.dims == [2, 3]
    check m[1, 1] == 0.5

  test "tensor":
    debug t1
    check t1.len == 24
    check t1.shape == arrayShape(F32, 2, 3, 4)
    check t1[1, 2, 3] == 24

  test "clone":
    var t2 = t1.clone
    t2[0, -1, 0] = 999'f32
    debug "t1 = ", t1
    debug "t2 = ", t2
    check t1[0, 2, 0] == 9
    check t2[0, 2, 0] == 999

  test "boolean":
    let t = [true, false, false, true].toTensor.reshape(2, 2)
    debug t
    check t[1, 0] == false
    check typeOf(t) is Tensor[bool]

  test "random":
    setPrintOpts(floatMode=ffDecimal, precision=4)
    let t = newSeqWith(20, rand(1.0)).toTensor.reshape(4, 5)
    debug t
    setPrintOpts()

  test "formatting":
    let t = toTensor[float32](1..10000).reshape(100, 100)
    let str = $t
    debug str
    check str.split("\n").len == 10

