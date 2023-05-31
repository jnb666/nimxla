{.warning[BareExcept]:off.}
import std/[unittest,logging]
import nimxla
import nimxla/[graph, tensor]

const debug {.booldefine.} = false
const gpu {.booldefine.} = false


suite "grad":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  if not debug:
    setLogLevel(Warning)
  when gpu:
    let client = newGPUClient()
  else:
    let client = newCPUClient()

  test "grad1":
    let b = newBuilder("grad1")
    let x = b.parameter(0, F64, name="x")
    let y = b.parameter(1, F64, name="y")
    let fwd = x * (x + y)
    let output = b.gradient(fwd, ["x", "y"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(4.0), lit(3.0)]).toLiterals
    debug "forward:", res[0]
    debug "grad(x):", res[1]
    debug "grad(y):", res[2]
    check res[0].f64[] == 28.0
    check res[1].f64[] == 11.0
    check res[2].f64[] == 4.0

  test "grad2":
    let g = newBuilder("grad2")
    let a = g.parameter(0, F64, name="a")
    let b = g.parameter(1, F64, name="b")
    let fwd = (a/b - a) * (b/a + a + b) * (a-b)
    let output = g.gradient(fwd, ["a", "b"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(230.3), lit(33.2)]).toLiterals
    debug "forward:", res[0]
    debug "grad(x):", res[1]
    debug "grad(y):", res[2]
    proc fn(a, b: float): float =
      (a/b - a) * (b/a + a + b) * (a-b)
    check res[0].f64[] == fn(230.3, 33.2)
    check abs(res[1].f64[] - -153284.83150602) < 1e-8
    check abs(res[2].f64[] - 3815.03894415) < 1e-8

  test "sigmoid":
    let g = newBuilder("sigmoid")
    let x = g.parameter(0, F64, name="x")
    let fwd = g.one(F64) / (g.one(F64) + exp(-x))
    let output = g.gradient(fwd, ["x"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(0.1)]).toLiterals
    debug "forward:", res[0]
    debug "grad(x):", res[1]
    check abs(res[0].f64[] - 0.52497919) < 1e-8
    check abs(res[1].f64[] - 0.24937604) < 1e-8

  test "sigmoid2":
    let g = newBuilder("sigmoid")
    let x = g.parameter(0, F64, name="x")
    let output = g.gradient(logistic(x), ["x"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(0.1)]).toLiterals
    debug "forward:", res[0]
    debug "grad(x):", res[1]
    check abs(res[0].f64[] - 0.52497919) < 1e-8
    check abs(res[1].f64[] - 0.24937604) < 1e-8

  test "sum":
    let g = newBuilder("sum")
    let x = g.parameter(0, F32, [3, 3], "x")
    let y = g.parameter(1, F32, [3, 3], "y")
    let fwd = reduceSum(x * y)
    let output = g.gradient(fwd, ["x", "y"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let xval = toTensor[float32](0..8).reshape(3, 3)
    let yval = fill([3, 3], 2f32)
    let res = exec.runAndUnpack([xval.toLiteral, yval.toLiteral]).toLiterals
    debug "forward:", res[0]
    check res[0].f32[] == 72
    debug "grad(x):", res[1]
    debug "grad(y):", res[2]
    check res[1].f32.toSeq == yval.toSeq
    check res[2].f32.toSeq == xval.toSeq

  test "broadcast":
    let g = newBuilder("broadcast")
    let x = g.parameter(0, F32, [], "x")
    let y = g.parameter(1, F32, [2, 2], "y")
    let fwd = x * (x + y)
    let output = g.gradient(fwd, ["x", "y"])
    let comp = build(output)
    debug comp
    let exec = client.compile(comp)
    let yval = fill([2, 2], 3f32)
    let res = exec.runAndUnpack([lit(4f32), yval.toLiteral]).toLiterals
    debug "forward:", res[0]
    check res[0].f32.toSeq == [28f32, 28, 28, 28]
    debug "grad(x):", res[1]
    debug "grad(y):", res[2]
    check res[1].shape == x.shape
    check res[1].f32[] == 44
    check res[2].shape == y.shape
    check res[2].f32.toSeq == [4f32, 4, 4, 4]

  test "matmul":
    proc test_matmul(x, y, fwd, gradX, gradY: Tensor[float64]) =
      let b = newBuilder("broadcast")
      let xp = b.parameter(0, F64, x.dims, "x")
      let yp = b.parameter(1, F64, y.dims, "y")
      let output = b.gradient(dot(xp, yp), ["x", "y"])
      let comp = build(output)
      debug comp
      let exec = client.compile(comp)
      let res = exec.runAndUnpack([x.toLiteral, y.toLiteral]).toLiterals
      debug "forward:", res[0]
      debug "grad(x):", res[1]
      debug "grad(y):", res[2]
      check res[0].f64 == fwd
      check res[1].f64 == gradX
      check res[2].f64 == gradY

    test_matmul(
      [2.0, 2.0, 2.0, 2.0].toTensor,
      [3.0, 3.0, 3.0, 3.0].toTensor,
      24.0.toTensor, 
      [3.0, 3.0, 3.0, 3.0].toTensor,
      [2.0, 2.0, 2.0, 2.0].toTensor
    )
    test_matmul(
      [2.0, 2.0, 2.0, 2.0, 
       3.0, 3.0, 3.0, 3.0].toTensor.reshape(2, 4),
      [3.0, 3.0, 3.0, 3.0].toTensor, 
      [24.0, 36.0].toTensor, 
      [3.0, 3.0, 3.0, 3.0, 
       3.0, 3.0, 3.0, 3.0].toTensor.reshape(2, 4),
      [5.0, 5.0, 5.0, 5.0].toTensor, 
    )
    test_matmul(
      [2.0, 2, 2, 2, 
       3.0, 3, 3, 3].toTensor.reshape(2, 4),
      [1.0, 2, 3, 4].toTensor.reshape(4, 1),
      [20.0, 30.0].toTensor.reshape(2, 1), 
      [1.0, 2, 3, 4, 
       1.0, 2, 3, 4].toTensor.reshape(2, 4),
      [5.0, 5, 5, 5].toTensor.reshape(4, 1)
    )


