{.warning[BareExcept]:off.}
import std/[unittest, logging, sequtils, math]
import nimxla

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
    let x = b.parameter(F64, name="x")
    let y = b.parameter(F64, name="y")
    let fwd = x * (x + y)
    let grads = b.gradient(fwd, ["x", "y"])
    let comp = b.build(b.makeTuple(grads))
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(4.0), lit(3.0)]).toLiterals
    debug "grad(x):", res[0]
    debug "grad(y):", res[1]
    check res[0].f64[] == 11.0
    check res[1].f64[] == 4.0

  test "grad2":
    let g = newBuilder("grad2")
    let a = g.parameter(F64, name="a")
    let b = g.parameter(F64, name="b")
    let fwd = (a/b - a) * (b/a + a + b) * (a-b)
    let grads = g.gradient(fwd, ["a", "b"])
    let comp = g.build(g.makeTuple(grads))
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([lit(230.3), lit(33.2)]).toLiterals
    debug "grad(x):", res[0]
    debug "grad(y):", res[1]
    check abs(res[0].f64[] - -153284.83150602) < 1e-8
    check abs(res[1].f64[] - 3815.03894415) < 1e-8

  test "sigmoid":
    let g = newBuilder("sigmoid")
    let x = g.parameter(F64, name="x")
    let fwd = g.one(F64) / (g.one(F64) + exp(-x))
    let grad = g.gradient(fwd, ["x"])
    let comp = g.build(grad[0])
    debug comp
    let exec = client.compile(comp)
    let res = exec.run([lit(0.1)]).toLiteral
    debug "grad(x):", res
    check abs(res.f64[] - 0.24937604) < 1e-8

  test "sigmoid2":
    let g = newBuilder("sigmoid")
    let x = g.parameter(F64, name="x")
    let grad = g.gradient(sigmoid(x), ["x"])
    let comp = g.build(grad[0])
    debug comp
    let exec = client.compile(comp)
    let res = exec.run([lit(0.1)]).toLiteral
    debug "grad(x):", res
    check abs(res.f64[] - 0.24937604) < 1e-8

  test "sum":
    let g = newBuilder("sum")
    let x = g.parameter(F32, [3, 3], "x")
    let y = g.parameter(F32, [3, 3], "y")
    let fwd = sum(x * y)
    debug fwd.repr
    let grads = g.gradient(fwd, ["x", "y"])
    let comp = g.build(g.makeTuple(grads))
    debug comp
    let exec = client.compile(comp)
    let xval = toTensor[float32](0..8).reshape(3, 3)
    let yval = fill([3, 3], 2f32)
    let res = exec.runAndUnpack([xval.toLiteral, yval.toLiteral]).toLiterals
    debug "grad(x):", res[0]
    debug "grad(y):", res[1]
    check res[0].f32.toSeq == yval.toSeq
    check res[1].f32.toSeq == xval.toSeq

  test "sum2":
    let g = newBuilder("sum")
    let x = g.parameter(F32, [3, 3], "x")
    let fwd = sum(x, [1])
    debug fwd.repr
    let grads = g.gradient(fwd, ["x"])
    let comp = g.build(grads[0])
    debug comp
    let exec = client.compile(comp)
    let xval = toTensor[float32](0..8).reshape(3, 3)
    let res = exec.run([xval.toLiteral]).toLiteral
    debug "grad(x):", res
    check res.f32.shape == xval.shape
    check res.f32.toSeq == newSeqWith(9, 1f32)

  test "broadcast":
    let g = newBuilder("broadcast")
    let x = g.parameter(F32, [], "x")
    let y = g.parameter(F32, [2, 2], "y")
    let fwd = x * (x + y)
    let grads = g.gradient(fwd, ["x", "y"])
    let comp = g.build(g.makeTuple(grads))
    debug comp
    let exec = client.compile(comp)
    let yval = fill([2, 2], 3f32)
    let res = exec.runAndUnpack([lit(4f32), yval.toLiteral]).toLiterals
    debug "grad(x):", res[0]
    debug "grad(y):", res[1]
    check res[0].shape == x.shape
    check res[0].f32[] == 44
    check res[1].shape == y.shape
    check res[1].f32.toSeq == [4f32, 4, 4, 4]

  test "matmul":
    proc test_matmul(x, y, gradX, gradY: Tensor[float64]) =
      let b = newBuilder("matmul")
      let xp = b.parameter(F64, x.dims, "x")
      let yp = b.parameter(F64, y.dims, "y")
      let grads = b.gradient(dot(xp, yp), ["x", "y"])
      let comp = b.build(b.makeTuple(grads))
      debug comp
      let exec = client.compile(comp)
      let res = exec.runAndUnpack([x.toLiteral, y.toLiteral]).toLiterals
      debug "grad(x):", res[0]
      debug "grad(y):", res[1]
      check res[0].f64 == gradX
      check res[1].f64 == gradY

    test_matmul(
      @@[2.0, 2, 2, 2],
      @@[3.0, 3, 3, 3],
      @@[3.0, 3, 3, 3],
      @@[2.0, 2, 2, 2]
    )
    test_matmul(
      @@[[2.0, 2, 2, 2], [3, 3, 3, 3]],
      @@[3.0, 3, 3, 3],
      @@[[3.0, 3, 3, 3], [3, 3, 3, 3]],
      @@[5.0, 5, 5, 5]
    )
    test_matmul(
      @@[[2.0, 2, 2, 2], [3, 3, 3, 3]],
      @@[1.0, 2, 3, 4],
      @@[[1.0, 2, 3, 4], [1, 2, 3, 4]],
      @@[5.0, 5, 5, 5]
    )

  test "conv1d":
    let x = toTensor[float64](1..5).reshape(1, 5, 1)
    let w = toTensor[float64](1..3).reshape(3, 1, 1)
    let b = newBuilder("conv1d")
    let input = b.parameter(F64, x.dims, "input")
    let kernel = b.parameter(F64, w.dims, "kernel")
    let output = conv1d(input, kernel) * b^w.reshape(1, 3, 1)
    let grads = b.gradient(output, ["input", "kernel"])
    let comp = b.build(b.makeTuple(output, grads[0], grads[1]))
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([x.toLiteral, w.toLiteral]).toLiterals
    debug "conv1d(input, kernel): ", res[0]
    debug "grad(input):", res[1]
    debug "grad(kernel):", res[2]
    check res[0].f64 == @@[[[14.0], [40], [78]]]
    check res[1].f64 == @@[[[1.0], [4], [10], [12], [9]]]
    check res[2].f64 == @@[[[14.0]], [[20]], [[26]]]

  test "conv2d":
    var t = newTensor[float64](1, 5, 5, 2)
    for y in 0..4:
      for x in 0..4:
        t[0, y, x, 0] = float(y*5 + x)
        t[0, y, x, 1] = t[0, y, x, 0] * 0.001
    debug "input = ", t
    var w = newTensor[float64](3, 3, 2, 1)
    for y in 0..2:
      for x in 0..2:
        w[y, x, 0, 0] = float(y*3 + x)
        w[y, x, 1, 0] = w[y, x, 0, 0]
    debug "kernel = ", w
    let b = newBuilder("conv2d")
    let input = b.parameter(F64, t.dims, "input")
    let kernel = b.parameter(F64, w.dims, "kernel")
    let y = conv2d(input, kernel)
    let grads = b.gradient(y, ["input", "kernel"])
    let comp = b.build(b.makeTuple(y, grads[0], grads[1]))
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([t.toLiteral, w.toLiteral]).toLiterals
    debug "conv2d(input, kernel): ", res[0]
    debug "grad(input):", res[1]
    debug "grad(kernel):", res[2]
    check res[0].f64.approxEqual(@@[[
      [[312.312], [348.348], [384.384]],
      [[492.492], [528.528], [564.564]],
      [[672.672], [708.708], [744.744]]]])
    check res[1].f64 == @@[[
      [[0.0, 0], [ 1,  1], [3,   3], [ 3,  3], [ 2,  2]],
      [[3, 3],   [ 8,  8], [15, 15], [12, 12], [ 7,  7]],
      [[9, 9],   [21, 21], [36, 36], [27, 27], [15, 15]],
      [[9, 9],   [20, 20], [33, 33], [24, 24], [13, 13]],
      [[6, 6],   [13, 13], [21, 21], [15, 15], [ 8,  8]],
    ]]
    check res[2].f64.approxEqual(@@[
      [[[54.0], [0.054]], [[ 63], [0.063]], [[ 72], [0.072]]],
      [[[ 99],  [0.099]], [[108], [0.108]], [[117], [0.117]]],
      [[[144],  [0.144]], [[153], [0.153]], [[162], [0.162]]]])

  test "conv2d_grouped":
    var t = newTensor[float64](2, 5, 5, 2)
    for y in 0..4:
      for x in 0..4:
        t[0, y, x, 0] = float(y*5 + x)
        t[0, y, x, 1] = t[0, y, x, 0] * 0.001
        t[1, y, x, 0] = float(y*5 + x + 10)
        t[1, y, x, 1] = t[1, y, x, 0] * 0.001
    debug "input = ", t
    var w = newTensor[float64](3, 3, 1, 2)
    for y in 0..2:
      for x in 0..2:
        w[y, x, 0, 0] = float(y*3 + x)
        w[y, x, 0, 1] = w[y, x, 0, 0]
    debug "kernel = ", w
    let b = newBuilder("conv2d")
    let input = b.parameter(F64, t.dims, "input")
    let kernel = b.parameter(F64, w.dims, "kernel")
    let y = conv2d(input, kernel, padding=pad(1), strides=2, groups=2)
    let grads = b.gradient(y, ["input", "kernel"])
    let comp = b.build(b.makeTuple(y, grads[0], grads[1]))
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([t.toLiteral, w.toLiteral]).toLiterals
    debug "conv2d(input, kernel): ", res[0]
    debug "grad(input):", res[1]
    debug "grad(kernel):", res[2]
    check res[0].f64.approxEqual(@@[
      [[[ 88.0,  0.088], [  175,  0.175], [  136,  0.136]],
       [[  345,  0.345], [  528,  0.528], [  345,  0.345]],
       [[  232,  0.232], [  319,  0.319], [  184,  0.184]]],
      [[[  328,  0.328], [  505,  0.505], [  336,  0.336]],
       [[  615,  0.615], [  888,  0.888], [  555,  0.555]],
       [[  352,  0.352], [  469,  0.469], [  264,  0.264]]]])
    check res[1].f64 == @@[
      [[[4.0,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]],
       [[  8,  8], [16, 16], [ 8,  8], [16, 16], [ 8, 8]],
       [[  4,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]],
       [[  8,  8], [16, 16], [ 8,  8], [16, 16], [ 8, 8]],
       [[  4,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]]],
      [[[  4,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]],
       [[  8,  8], [16, 16], [ 8,  8], [16, 16], [ 8, 8]],
       [[  4,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]],
       [[  8,  8], [16, 16], [ 8,  8], [16, 16], [ 8, 8]],
       [[  4,  4], [ 8,  8], [ 4,  4], [ 8,  8], [ 4, 4]]]]
    check res[2].f64.approxEqual(@@[
      [[[136.0, 0.136]], [[204, 0.204]], [[136, 0.136]]],
      [[[  204, 0.204]], [[306, 0.306]], [[204, 0.204]]],
      [[[  136, 0.136]], [[204, 0.204]], [[136, 0.136]]]])

  test "max_pool1d":
    let x = @@[[[0.0, 0], [1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]]]
    let b = newBuilder("pool1d")
    let input = b.parameter(F64, x.dims, "input")
    let grads = b.gradient(maxPool1d(input, kernelSize=2, strides=1), ["input"])
    let comp = b.build(grads[0])
    debug comp
    let exec = client.compile(comp)
    let res = exec.run([x.toLiteral]).f64
    debug "grad(input):", res
    check res == @@[[[0.0, 0], [1, 1], [1, 1], [1, 1], [1, 1]]]

  test "max_pool2d":
    let x = toTensor[float64](0 .. 24).reshape(1, 5, 5, 1)
    let b = newBuilder("pool2d")
    let input = b.parameter(F64, x.dims, "input")
    let y = maxPool2d(input, kernelSize=2)
    let grads = b.gradient(y, ["input"])
    let comp = b.build b.makeTuple(y, grads[0])
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([x.toLiteral]).tuple2
    let yval = res[0].f64
    debug "y: ", yval
    check yval == @@[[[[6.0], [8]], [[16], [18]]]]
    let grad = res[1].f64
    debug "grad(input):", grad
    check grad == @@[[
      [[0.0], [0], [0], [0], [0]],
      [[0],   [1], [0], [1], [0]],
      [[0],   [0], [0], [0], [0]],
      [[0],   [1], [0], [1], [0]],
      [[0],   [0], [0], [0], [0]],
    ]]

  test "avg_pool2d":
    let x = toTensor[float64](0 .. 24).reshape(1, 5, 5, 1)
    let b = newBuilder("pool2d")
    let input = b.parameter(F64, x.dims, "input")
    let y = avgPool2d(input, kernelSize=2)
    let grads = b.gradient(y, ["input"])
    let comp = b.build b.makeTuple(y, grads[0])
    debug comp
    let exec = client.compile(comp)
    let res = exec.runAndUnpack([x.toLiteral]).tuple2
    let yval = res[0].f64
    debug "y: ", yval
    check yval == @@[[[[3.0], [5]], [[13], [15]]]]
    let grad = res[1].f64
    debug "grad(input):", grad
    check grad == @@[[
      [[0.25], [0.25], [0.25], [0.25], [0]],
      [[0.25], [0.25], [0.25], [0.25], [0]],
      [[0.25], [0.25], [0.25], [0.25], [0]],
      [[0.25], [0.25], [0.25], [0.25], [0]],
      [[0],   [0], [0], [0], [0]],
    ]]