{.warning[BareExcept]:off.}
import std/[unittest, logging, math]
import nimxla

const debug {.booldefine.} = false
const gpu {.booldefine.} = false

suite "client":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  if not debug:
    setLogLevel(Warning)
  when gpu:
    let client = newGPUClient()
  else:
    let client = newCPUClient()

  let t1 = toTensor[int32](1..24).reshape(2, 3, 4)

  test "scalar":
    let scalar = lit(3.142)
    debug scalar
    check scalar.len == 1
    check scalar.dtype == F64
    check scalar.f64[] == 3.142

  test "vector":
    let data = @[1f32, 2, 3, 4, 5, 6]
    let vec = data.toTensor.reshape(2, 3).toLiteral
    debug vec
    check vec.len == 6
    check vec.shape == arrayShape(F32, 2, 3)
    check vec.f32.toSeq == data
  
  test "copy1":
    let lit = t1.toLiteral
    let buf = client.newBuffer(lit)
    debug buf
    check buf.shape == t1.shape
    let lit2 = buf.toLiteral
    check toTensor[int32](lit2) == t1

  test "copy2":
    let buf = client.newBuffer(t1)
    debug buf
    check buf.shape == t1.shape
    let t2 = toTensor[int32](buf)
    check t2 == t1


