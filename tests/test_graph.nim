{.warning[BareExcept]:off.}
import std/[unittest, logging, strformat, math, sequtils, strutils]
import nimxla
import nimxla/[graph, tensor]

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
    let sum = b.constant(20.0) + b.constant([22.0, 44.0])
    check sum.shape == arrayShape(F64, 2)
    let comp = build(sum)
    debug comp
    let exec = client.compile(comp)
    let res = toTensor[float64](exec.run)
    debug "result = ", res
    check res.dims == [2]
    check res.toSeq == [42.0, 64.0]

  test "param":
    let b = newBuilder("getSqrt")
    let x = b.parameter(0, F32, name="x")
    let comp = build sqrt(x)
    debug comp
    check comp.params.len == 1
    check comp.params[0].name == "x"
    let exec = client.compile(comp)
    for x in 1 .. 5:
      let res = exec.run([lit(x.float32)]).toLiteral.f32
      debug &"sqrt({x}) = {res}" 
      check res.dims == []
      check abs(res[] - sqrt(x.float32)) < 1e-6

  test "param2":
    let b = newBuilder("hypotenuse")
    let x = b.parameter(0, F32, name="x")
    let y = b.parameter(1, F32, name="y")
    let comp = build sqrt(x*x + y*y)
    debug comp
    check comp.paramNames == ["x", "y"]
    let exec = client.compile(comp)
    for i in 1 .. 3:
      for j in 2 .. 4:
        let x = i.float32
        let y = j.float32
        let res = exec.run([lit(x), lit(y)]).toLiteral.f32
        debug &"hypot({x}, {y}) = {res}" 
        check res.dims == []
        check abs(res[] - sqrt(x*x + y*y)) < 1e-6

  test "reduce_sum":
    let vec = toTensor[float32](1..12).toLiteral
    debug vec
    let b = newBuilder("reduce")
    let a = b.parameter(0, F32, [12])
    let comp = build(a.reduceSum)
    debug comp
    let res = client.compile(comp).run([vec]).toLiteral.f32
    debug res
    check res.dims == []
    check res[] == 6.0*13.0

  test "reduce_max":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("reduce")
    let a = b.parameter(0, I32, [3, 4])
    let comp = build(a.reduceMax([1]))
    debug comp
    let res = client.compile(comp).run([t1]).toLiteral.i32
    debug res
    check res.dims == [3]
    check res.toSeq == [4'i32, 8, 12]

  test "reduce_max2":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("reduce")
    let a = b.parameter(0, I32, [3, 4])
    let comp = build(a.reduceMax([1], keepDims=true))
    debug comp
    let res = client.compile(comp).run([t1]).toLiteral.i32
    debug res
    check res.dims == [3, 1]
    check res.toSeq == [4'i32, 8, 12]

  test "random":
    setPrintOpts(floatMode=ffDecimal, precision=4)
    let b = newBuilder("test")
    let comp = rngUniform(b.zero(BF16), b.one(BF16), [5, 10]).build
    debug comp
    let exec = client.compile(comp)
    for _ in 1 .. 5:
      let lit = exec.run.toLiteral
      debug $lit
      check lit.shape == arrayShape(BF16, 5, 10)
    setPrintOpts()

  test "transpose":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(0, I32, [3, 4])
    let comp = a.transpose.build
    debug comp
    let res = client.compile(comp).run([t1]).toLiteral.i32
    debug res
    check res.dims == [4, 3]
    check res.toSeq == [1'i32, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

  test "narrow":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(0, I32, [3, 4])
    let comp = a.narrow(1, 1, 3).build
    debug comp
    let res = client.compile(comp).run([t1]).toLiteral.i32
    debug res
    check res.dims == [3, 2]
    check res.toSeq == [2i32, 3, 6, 7, 10, 11]

  test "compare":
    let t1 = toTensor[int32](1..12).reshape(3, 4).toLiteral
    debug t1
    let b = newBuilder("test")
    let a = b.parameter(0, I32, [3, 4])
    let comp = build(!logicalAnd(a > b.constant(2, I32), a < b.constant(10, I32)))
    debug comp
    let res = toTensor[bool](client.compile(comp).run([t1]))
    debug res
    check res.shape == arrayShape(Bool, 3, 4)
    check count(res.toSeq, true) == 5

  test "tuple":
    # using run to return the tuple without unpacking
    let b = newBuilder("test")
    let comp = build(b.makeTuple(b.constant(1f32), b.constant([2f32, 3])))
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
    let comp = build(b.makeTuple(b.constant(1f32), b.constant([2f32, 3])))
    debug comp
    let res = client.compile(comp).runAndUnpack
    debug res
    check res.len == 2
    check res[0].shape == arrayShape(F32)
    check res[0].toLiteral.f32[] == 1
    check res[1].shape == arrayShape(F32, 2)
    check res[1].toLiteral.f32.toSeq == [2f32, 3]

  test "matmul":
    let b = newBuilder("test")
    let x = b.parameter(0, F32, [2, 3], "x")
    let y = b.parameter(1, F32, [3, 2], "y")
    let comp = build dot(x, y)
    let mx = toTensor[float32](1..6).reshape(2, 3).toLiteral
    let my = toTensor[float32](7..12).reshape(3, 2).toLiteral
    debug "x = ", mx
    debug "y = ", my
    let res = client.compile(comp).run([mx, my]).toLiteral.f32
    debug "dot(x, y) = ", res
    check res.dims == [2, 2]
    check res.toSeq == [58f32, 64, 139, 154]

  test "error":
    let b = newBuilder("test")
    let x = b.parameter(0, F32, [2, 5], "x")
    let y = b.parameter(1, F32, [3, 2], "y")
    try:
      let sum = dot(x, y + b.one(F32)) / b.constant(10f32)
    except BuilderError as e:
      debug &"got error: '{e.origMsg}' at\n{e.at.repr}"
      check e.origMsg == "Cannot infer shape for dot operation: f32[2,5] <dot> f32[3,2]. Contracting dimension sizes do not match."
      check e.at.kind == tDot
      check e.at[0].shape.dims == [2, 5]
      check e.at[1].shape.dims == [3, 2]

  test "check_shape":
    let b = newBuilder("test")
    let foo = b.parameter(0, I32, [3, 4], "foo")
    let comp = build foo * foo
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