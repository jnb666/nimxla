{.warning[BareExcept]:off.}
import std/[unittest, logging, strformat, math, sequtils, strutils]
import nimxla

const debug {.booldefine.} = false
const gpu {.booldefine.} = false

suite "graph":
  var logger = newConsoleLogger(levelThreshold=if debug: lvlDebug else: lvlInfo)
  addHandler(logger)
  if not debug:
    setLogLevel(Warning)
  when gpu:
    let client = newGPUClient()
  else:
    let client = newCPUClient()

  test "add":
    let b = newBuilder("test")
    let sum = b^20.0 + b^[22.0, 44.0]
    check sum.shape == arrayShape(F64, 2)
    let comp = b.build(sum)
    debug comp
    let exec = client.compile(comp)
    let res = exec.run.f64
    debug "result = ", res
    check res.dims == [2]
    check res.toSeq == [42.0, 64.0]

  test "addInt":
    let b = newBuilder("test")
    let sum = b^20 + b^[22, 44]
    check sum.shape == arrayShape(I64, 2)
    let comp = b.build(sum)
    debug comp
    let exec = client.compile(comp)
    let res = exec.run.i64
    debug "result = ", res
    check res.dims == [2]
    check res.toSeq == [42i64, 64]

  test "param":
    let b = newBuilder("getSqrt")
    let x = b.parameter(F32, name="x")
    let comp = b.build sqrt(x)
    debug comp
    check comp.params.len == 1
    check comp.params[0].name == "x"
    let exec = client.compile(comp)
    for x in 1 .. 5:
      let res = exec.run([lit(x.float32)]).f32
      debug &"sqrt({x}) = {res}" 
      check res.dims == []
      check abs(res[] - sqrt(x.float32)) < 1e-6

  test "param2":
    let b = newBuilder("hypotenuse")
    let x = b.parameter(F32, name="x")
    let y = b.parameter(F32, name="y")
    let comp = b.build sqrt(x*x + y*y)
    debug comp
    check comp.paramNames == ["x", "y"]
    let exec = client.compile(comp)
    for i in 1 .. 3:
      for j in 2 .. 4:
        let x = i.float32
        let y = j.float32
        let res = exec.run([lit(x), lit(y)]).f32
        debug &"hypot({x}, {y}) = {res}" 
        check res.dims == []
        check abs(res[] - sqrt(x*x + y*y)) < 1e-6

  test "reduce_sum":
    let vec = toTensor[float32](1..12).toLiteral
    debug vec
    let b = newBuilder("reduce")
    let a = b.parameter(F32, [12])
    let comp = b.build(a.sum)
    debug comp
    let res = client.compile(comp).run([vec]).f32
    debug res
    check res.dims == []
    check res[] == 6.0*13.0

  test "reduce_max":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("reduce")
    let a = b.parameter(I32, [3, 4])
    let comp = b.build(a.max(axis=1))
    debug comp
    let res = client.compile(comp).run([t1]).i32
    debug res
    check res.dims == [3]
    check res.toSeq == [4'i32, 8, 12]

  test "reduce_mean":
    let t1 = toTensor[float32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("reduce")
    let a = b.parameter(F32, [3, 4])
    let comp = b.build(a.mean(axis=1, keepDims=true))
    debug comp
    let res = client.compile(comp).run([t1]).f32
    debug res
    check res.dims == [3, 1]
    check res.toSeq == [2.5f32, 6.5, 10.5]

  test "argmax":
    let t1 = @@[[7f32,  3,  9,   4 ],
                [   2, -1, -2,   0 ],
                [ 0.5,  7,  3, 4.2 ]].toLiteral
    debug t1
    let b = newBuilder("reduce")
    let a = b.parameter(F32, [3, 4])
    let comp = b.build(a.argmax(0))
    debug comp
    let res = client.compile(comp).run([t1]).i64
    debug res
    check res.dims == [4]
    check res.toSeq == [0i64, 2, 0, 2]

  test "iota":
    let b = newBuilder("test")
    let comp = b.build b.iota(F32, [4, 8], 1)
    debug comp
    let exec = client.compile(comp)
    let lit = exec.run.toLiteral
    debug lit
    check lit.shape == arrayShape(F32, 4, 8)
    check lit.f32[0, 0] == 0.0
    check lit.f32[3, 7] == 7.0

  test "random":
    setPrintOpts(floatMode=ffDecimal, precision=4)
    let b = newBuilder("test")
    let comp = b.build rngUniform(b.zero(BF16), b.one(BF16), [5, 10])
    debug comp
    let exec = client.compile(comp)
    for _ in 1 .. 5:
      let lit = exec.run.toLiteral
      debug lit
      check lit.shape == arrayShape(BF16, 5, 10)
    setPrintOpts()

  test "transpose":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(I32, [3, 4])
    let comp = b.build a.transpose
    debug comp
    let res = client.compile(comp).run([t1]).i32
    debug res
    check res.dims == [4, 3]
    check res.toSeq == [1'i32, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

  test "reverse":
    let t1 = toTensor[int32](1..12).reshape(2, 2, 3)
    debug t1
    let b = newBuilder("test")
    let comp = b.build reverse(b^t1, 2)
    debug comp
    let res = client.compile(comp).run.i32
    debug res
    check res.dims == [2, 2, 3]
    check res.toSeq == [3'i32, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]

  test "narrow":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(I32, [3, 4])
    let comp = b.build a.narrow(1, 1, 3)
    debug comp
    let res = client.compile(comp).run([t1]).i32
    debug res
    check res.dims == [3, 2]
    check res.toSeq == [2i32, 3, 6, 7, 10, 11]

  test "concat":
    let b = newBuilder("test")
    let comp = b.build concat(b.constant(@@[[1, 2, 3]]), [b.constant(@@[[4, 5, 6]]), b.constant(@@[[7, 8, 9]])], axis=0)
    let res = client.compile(comp).run.i64
    debug res
    check res == toTensor[int64](1..9).reshape(3, 3)

  test "gather":
    let t1 = toTensor[float32](1..12).reshape(2, 6)
    debug "tensor = ", t1

    proc test_gather(ix: Tensor[int64], expect: Tensor[float32]) =
      debug "ix = ", ix
      let b = newBuilder("test")
      let a = b.parameter(F32, t1.dims)
      let index = b.parameter(I64, ix.dims)
      let comp = b.build a.gather(index)
      let res = client.compile(comp).run([t1.toLiteral, ix.toLiteral]).f32
      debug res
      check res == expect

    test_gather(
      @@[[1, 1], [0, 2], [1, 5]], @@[8f32, 3, 12])
    test_gather(
      @@[[[0, 3], [1, 3]], [[1, 0], [0, 0]]], @@[[4f32, 10], [7, 1]])

  test "add_at":
    let t1 = toTensor[float32](0..5).reshape(1, 2, 3)
    debug "tensor = ", t1
    let ix = @@[[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 2], [0, 1, 2]]
    debug "ix = ", ix
    let update = @@[1f32, 3, 1, -2, 5]
    debug "update = ", update
    let b = newBuilder("test")
    let comp = b.build addAt(b^t1, b^ix, b^update)
    let res = client.compile(comp).run.f32
    debug res
    check res == @@[[[2f32, 1, 0], [3, 7, 10]]]

  test "compare":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(I32, [3, 4])
    let comp = b.build !logicalAnd(a > b^2i32, a < b^10i32)
    debug comp
    let res = toTensor[bool](client.compile(comp).run([t1]))
    debug res
    check res.shape == arrayShape(Bool, 3, 4)
    check count(res.toSeq, true) == 5

  test "tuple":
    # using run to return the tuple without unpacking
    let b = newBuilder("test")
    let comp = b.build b.makeTuple(b^1f32, b^[2f32, 3])
    debug comp
    let lit = client.compile(comp).run.toLiteral
    debug lit
    let s = lit.shape
    check s.kind == TupleKind
    check s.elems == [arrayShape(F32), arrayShape(F32, 2)]
    let items = lit.decomposeTuple
    check items.len == 2
    check items[0].f32.toSeq == [1f32]
    check items[1].f32.toSeq == [2f32, 3]

  test "tuple2":
    # using runAll to unpack into a list of buffers
    let b = newBuilder("test")
    let comp = b.build b.makeTuple(b^1f32, b^[2f32, 3])
    debug comp
    let (v1, v2) = client.compile(comp).runAndUnpack.tuple2
    debug v1, " ", "v2"
    check v1.shape == arrayShape(F32)
    check v1.f32[] == 1
    check v2.shape == arrayShape(F32, 2)
    check v2.f32.toSeq == [2f32, 3]

  test "matmul":
    let b = newBuilder("test")
    let x = b.parameter(F32, [2, 3], "x")
    let y = b.parameter(F32, [3, 2], "y")
    let comp = b.build dot(x, y)
    let mx = toTensor[float32](1..6).reshape(2, 3).toLiteral
    let my = toTensor[float32](7..12).reshape(3, 2).toLiteral
    debug "x = ", mx
    debug "y = ", my
    let res = client.compile(comp).run([mx, my]).f32
    debug "dot(x, y) = ", res
    check res.dims == [2, 2]
    check res.toSeq == [58f32, 64, 139, 154]

  test "pad":
    let b = newBuilder("test")
    let x = toTensor[float32](1..6).reshape(2, 3)
    let comp = b.build pad(b^x, b.zero(F32), [(1, 1, 0), (1, 1, 0)])
    let res = client.compile(comp).run.f32
    debug res
    check res == @@[
      [0f32, 0, 0, 0, 0],
      [   0, 1, 2, 3, 0],
      [   0, 4, 5, 6, 0],
      [   0, 0, 0, 0, 0]
    ]

  test "one_hot":
    let b = newBuilder("test")
    let x = [1i32, 0, 3, 4].toTensor
    let comp = b.build oneHot(b^x, 5, F32)
    let res = client.compile(comp).run.f32
    debug res
    check res == @@[
      [0f32, 1, 0, 0, 0],
      [   1, 0, 0, 0, 0],
      [   0, 0, 0, 1, 0],
      [   0, 0, 0, 0, 1]
    ]

  test "conv2d":
    proc test_conv(dims: openarray[int], axis: int, scale: float32,
                   kernel, expect: Tensor[float32], padding = pad(0),
                   strides = 1, dilations = 1, channelsFirst = false) =
      let b = newBuilder("test")
      debug &"strides={strides} padding={padding} dilations={dilations} channelsFirst={channelsFirst}"
      let x = b.iota(F32, dims, axis)
      let x2 = x.concat([x * b^scale], axis = if channelsFirst: 1 else: -1)
      let comp = b.build conv2d(x2, b^kernel, strides, padding, dilations, channelsFirst=channelsFirst)
      let res = client.compile(comp).run.f32
      debug res
      check res.approxEqual(expect)

    test_conv(
      [1, 3, 3, 1], 2, 0.1,
      fill([1, 3, 3, 2], 1f32),
      @@[[[[9.9f32]]]],
    )
    test_conv(
      [1, 1, 3, 3], 2, 0.1,
      fill([1, 2, 3, 3], 1f32),
      @@[[[[9.9f32]]]],
      channelsFirst=true
    )
    test_conv(
      [1, 1, 3, 3], 2, 0.1,
      fill([1, 2, 3, 3], 1f32),
      @@[[[[2.2f32, 3.3, 2.2], [6.6, 9.9, 6.6], [6.6, 9.9, 6.6]]]],
      padding=padSame, channelsFirst=true
    )
    test_conv(
      [1, 3, 3, 1], 1, 0.1,
      fill([1, 2, 2, 2], 1f32),
      @@[[
        [[2.2f32], [2.2], [1.1]],
        [[6.6], [6.6], [3.3]],
        [[4.4], [4.4], [2.2]]
      ]],
      padding=padSame,
    )
    test_conv(
      [1, 3, 3, 1], 2, 0.1,
      fill([1, 3, 3, 2], 1f32),
      @@[[[[2.2f32], [6.6]], [[2.2], [6.6]]]],
      padding=padSame, strides=2
    )
    test_conv(
      [1, 5, 5, 1], 1, 0.01,
      fill([1, 3, 3, 2], 1f32),
      @@[[[[18.18f32]]]],
      dilations=2
    )

  test "max_pool":
    proc test_pool(dims: openarray[int], expect: Tensor[float32],
                   window: int, strides = 0, padding = pad(0), channelsFirst = false) =
      let b = newBuilder("test")
      debug &"window={window} strides={strides} padding={padding} channelsFirst={channelsFirst}"
      let x = b.iota(F32, dims, axis=2)
      let x2 = x.concat([x * b^0.1f32], axis = if channelsFirst: 1 else: -1)
      let comp = b.build maxPool2d(x2, window, strides, padding, channelsFirst=channelsFirst)
      let res = client.compile(comp).run.f32
      debug res
      check res == expect

    test_pool(
      [1, 3, 3, 1],
      @@[[[[2f32, 0.2]]]],
      window=3
    )
    test_pool(
      [1, 1, 3, 3],
      @@[[[[2f32]], [[0.2]]]],
      window=3, channelsFirst=true
    )
    test_pool(
      [1, 1, 3, 3],
      @@[[[[1f32,   1,   1], [  2,   2,   2], [  2,   2,   2]],
          [[0.1 , 0.1, 0.1], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]
        ]],
      window=3, strides=1, padding=padSame, channelsFirst=true
    )

  test "avg_pool":
    let b = newBuilder("test")
    let x = b.iota(F32, [1, 3, 3, 1], axis = 2)
    let x2 = x.concat([x * b^0.1f32], axis = -1)
    let comp = b.build avgPool2d(x2, 3)
    let res = client.compile(comp).run.f32
    debug res
    check res == @@[[[[1f32, 0.1]]]]

  test "batch_norm":
    let b = newBuilder("test")
    let input = b.iota(F32, [7, 3], axis=0)
    let scale = b^[1f32, 2, 3]
    let offset = b^[10f32, 100, 1000]
    let mean = b^[0.5f32, 0.5, 1]
    let variance = b^[1f32, 1, 10]
    let comp = b.build batchNormInference(input, scale, offset, mean, variance, 1e-7, -1)
    let res = client.compile(comp).run.f32
    debug res
    check res.approxEqual(@@[
      [9.5f32, 99, 999.05133],
      [10.5,  101, 1000],
      [11.5,  103, 1000.94867],
      [12.5,  105, 1001.8974],
      [13.5,  107, 1002.84607],
      [14.5,  109, 1003.79474],
      [15.5,  111, 1004.7434]
    ])

  test "error":
    let b = newBuilder("test")
    let x = b.parameter(F32, [2, 5], "x")
    let y = b.parameter(F32, [3, 2], "y")
    try:
      let sum = dot(x, y + b.one(F32)) / b.constant(10, F32)
    except BuilderError as e:
      debug &"got error: '{e.origMsg}' at\n{e.at.repr}"
      check e.origMsg == "Cannot infer shape for dot operation: f32[2,5] <dot> f32[3,2]. Contracting dimension sizes do not match."
      check e.at.kind == tDot
      check e.at.args[0].dims == [2, 5]
      check e.at.args[1].dims == [3, 2]

  test "check_shape":
    let b = newBuilder("test")
    let foo = b.parameter(I32, [3, 4], "foo")
    let comp = b.build foo * foo
    let input = toTensor[int32](1..12).toLiteral
    let exec = client.compile(comp)
    try:
      let res = exec.run([input])
    except XLAError as e:
      debug &"got error: '{e.msg}'"
      check e.msg == "test.3: foo shape should be <i32 3 4> - got <i32 12>"
    let res = exec.run([input], checkShape=false)
    debug res.toLiteral
    check res.shape == arrayShape(I32, 3, 4)

