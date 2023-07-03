## The graph module wraps the XLA API in order to build and compile a graph of operations.
##
## A graph is a tree of Nodes (which each wrap an XLA operation). The Builder is used to construct new Nodes
## and to finalize the resulting graph to a Computation.
##
## The nodes in the graph are typed - i.e. each has a shape which defines the data type and dimensions.
## Each node will derive it's shape from it's inputs or throw an XLAError exception at build time if they are
## not compatible. See https://www.tensorflow.org/xla/operation_semantics for more details.
##
## If there is an error during construction of the graph an BuilderError exception is raised with
## the details of the current AST tree and the reason for the error.
##
## A computation can be compiled for a specific device using the Client which is defined using the nimxla module.
## Shapes and host literal types are defined in the literal module.
##

import std/[sugar, sequtils, strformat, strutils, tables, math, macros, logging]
import tensor, literal, shape
import private/[xla_wrapper, utils]

export shape

type
  BuilderError* = ref object of CatchableError
    ## Exception raised while building the graph. origMsg is the original status message.
    ## at indicates the node in the graph where the error occured.
    ## The repr of this node is added to the msg field. 
    origMsg*: string
    at*:      Node

  OpType* = enum
    tNone, tConst, tLiteral, tParam, tError, tIota,             ## leaf nodes
    tNot, tNeg, tAbs, tExp, tFloor, tCeil, tRound, tLog,        ## 1 arg ops
    tLog1p, tSigmoid, tRelu, tSign, tCos, tSin, tTanh, tSqrt,   ## ..
    tRsqrt, tIsFinite, tCopy, tZerosLike, tTupleElement,        ## ..
    tReshape, tBroadcast, tBroadcastInDim, tCollapse,           ## ..
    tTranspose, tNarrow, tConvert, tReverse, tMaxPool,          ## ..
    tReduceSum, tReduceMin, tReduceMax, tArgmin, tArgmax,       ## ..
    tAdd, tSub, tMul, tDiv, tRem, tMax, tMin, tPow, tDot,       ## 2 arg ops
    tAnd, tOr, tEq, tNe, tGe, tGt, tLe, tLt, tRngUniform,       ## ..
    tRngNormal, tReduce, tGather, tConv, tReduceWindow,         ## ..
    tSelectAndScatter,                                          ## ..
    tSelect, tClamp, tTuple, tConcat, tScatter,                 ## 3 or more arg ops
    tBatchNormInference, tBatchNormTraining, tBatchNormGrad     ## ..

  BuilderObj = object
    c: xla_builder

  Builder* = ref object
    ## Builder is used to construct new operations. It holds a reference to a xla_builder.
    params: seq[Node]
    nodes:  seq[Node]
    obj: BuilderObj

  Op = object
    c:  xla_op

  Node* = ref object
    ## A Node is generated from each xla_op once it is added to the graph.
    ## The id number is the index to the nodes sequence in the Computation object and is set by the builder.
    ## The shape is the output data type and dimesnions for the op. This must be fixed and known at build time.
    ## If the noGrad attribute is set then gradients are not accumulated from this node or it's inputs.
    id*:      int
    shape*:   Shape
    args*:    seq[Node]
    noGrad*:  bool
    builder*: Builder
    case kind*: OpType:
    of tParam:
      name*:    string
      paramId:  int
    of tError:
      message:  string
    of tTupleElement, tConcat:
      index:    int
    of tReshape, tBroadcast, tCollapse, tTranspose, tNarrow, tRngUniform, tRngNormal, tReduce, tReduceSum, 
       tReduceMin, tReduceMax, tArgmin, tArgmax, tReverse:
      indices:  seq[int]
    of tBroadcastInDim:
      outSize, bcastDims: seq[int]
    of tReduceWindow, tMaxPool:
      wdims:    seq[int]
      wstride:  seq[int]
      wpad:     seq[Padding]
    of tConv:
      idims:    seq[int]
      odims:    seq[int]
      kdims:    seq[int]
      strides:  seq[int]
      padding:  seq[Padding]
      dilation: seq[int]
      groups:   int
    of tBatchNormInference, tBatchNormTraining, tBatchNormGrad:
      epsilon:  float32
      axis:     int
    else:
      discard
    info:   string
    op:     ref Op
    idnum:  uint64
    tupleGrads: seq[Node]

  ComputationObj = object
    c: xla_computation

  Computation* = object
    ## A Computation wraps the constructed graph after it has been finalised. It holds a reference to
    ## the xla_computation object.
    ##
    ## The nodes sequence is a distinct list of nodes in order in which they were declared.
    ## params contatins a reference the parameters indexed by the provided index when they were defined.
    nodes*:  seq[Node]
    params*: seq[Node]
    obj: ref ComputationObj

# memory management
proc `=copy`(a: var BuilderObj, b: BuilderObj) {.error.}
proc `=copy`(a: var ComputationObj, b: ComputationObj) {.error.}
proc `=copy`(dst: var Op, src: Op) {.error.}

proc `=destroy`(builder: var BuilderObj) =
  if builder.c != nil:
    trace "free Builder"
    xla_builder_free(builder.c)
    builder.c = nil

proc `=destroy`(comp: var ComputationObj) =
  if comp.c != nil:
    xla_computation_free(comp.c)
    trace "free Computation"
    comp.c = nil

proc `=destroy`(op: var Op) =
  if op.c != nil:
    trace "free Op"
    xla_op_free(op.c)
    op.c = nil

# error handling
proc repr*(n: Node): string

proc raiseError*(message: string, node: Node = nil) =
  ## Raise a BuilderError exception annotated with the repr for the given node.
  var err: BuilderError
  new err
  err.name = "BuilderError"
  err.origMsg = message
  err.msg = "\n" & message
  if node != nil:
    err.at = node
    err.msg &= " at\n" & node.repr & "\n"
  raise err

proc checkBuilderError(status: status_t, at: Node = nil) =
  ## Check status code returned from XLA. If non nil then raise a BuilderError exception. 
  if status != nil:
    let message = $status_error_message(status)
    status_free(status)
    raiseError(message, at)

proc i64seq(arr: openarray[int], length: int, def: int64 = 0, name = ""): seq[int64] =
  if arr.len == 0:
    result = repeat(def, length)
  elif arr.len == length:
    result = map(arr, x => x.int64)
  else:
    raiseError(&"convolution: number of {name} values should be {length} - got {arr.len}")

proc getShape(b: Builder, op: xla_op): (Shape, status_t) =
  ## Returns the shape of the output from the op.
  var s: shape_t
  let status = b.obj.c.get_shape(op, s.addr)
  if status == nil:
    (toShape(s), nil)
  else:
    (Shape(), status)

proc uid(n: Node): uint64 = 
  ## Unique node id from pointer to xla_op
  if n.op != nil:
    cast[uint64](n.op.c)
  else:
    n.idnum

proc dtype*(n: Node): DataType = 
  ## Element type for this node
  n.shape.dtype

proc rank*(n: Node): int =
  ## Number of dimensions in the node output.
  n.shape.dims.len

proc dims*(n: Node): seq[int] =
  ## Dimensions of this node output.
  n.shape.dims

proc len*(n: Node): int =
  ## Get number of inputs to the node
  n.args.len

proc `$`*(n: Node): string =
  ## Print node id, type, shape and info fields.
  if n.id > 0:
    result.add &"{n.id:3}: "
  result.add ($n.kind)[1..^1] & $n.shape & $n.info

proc repr*(n: Node): string =
  ## Formatted AST tree of this node and all of it's children.
  result = $n
  for arg in n.args:
    result.add "\n" & indent(arg.repr, 2)

proc toString*(n: Node): string =
  ## Node name and argument names, expanded
  if n == nil:
    return "<nil>"
  case n.kind
  of tConst, tLiteral, tParam:
    n.info
  else:
    var name = ($n.kind)[1 .. ^1]
    name.removePrefix("reduce")
    name[0] = name[0].toLowerAscii
    name & "(" & map(n.args, x => x.toString).join(", ") & ")"

proc newBuilder*(name: string): Builder =
  ## Create a new builder which is used to generate a new graph. The name is used for debug info.
  trace "new Builder"
  new result
  result.obj.c = xla_builder_create(name)

proc addNode(b: Builder, node: var Node): Node =
  ## If node already defined return it from the nodes list, else add it and assign it's id.
  for n in b.nodes:
    if n.uid == node.uid:
      node.op = nil
      return n
    if node.kind == tConst and n.kind == tConst and n.shape == node.shape and n.info == node.info:
      node.op = nil
      return n
  node.id = b.nodes.len + 1
  b.nodes.add node
  return node

proc wrap(b: Builder, xop: xla_op, typ: OpType, args: openarray[Node] = [], info = ""): Node =
  ## Wrap the raw xla_op pointer as a Node. Will throw an exception if there is an error e.g. incorrect shape.
  trace "new Op"
  var node = Node(builder: b, kind: typ, op: new Op, args: @args, info: info)
  node.op.c = xop
  var (s, err) = b.getShape(xop)
  if err == nil:
    node.shape = s
    err = b.obj.c.get_current_status
  checkBuilderError(err, node)
  b.addNode(node)

proc build*(b: Builder, root: Node): Computation =
  ## Build a computation from the specified root operation. Should only be called once for a given graph.
  trace "new Computation"
  result.obj = new ComputationObj
  let status = b.obj.c.build(root.op.c, result.obj.c.addr)
  checkBuilderError(status, root)
  for n in mitems(b.nodes):
    n.op = nil
  result.nodes = b.nodes
  result.params = b.params
  b.nodes = @[]
  b.params = @[]

proc last*(comp: Computation): Node =
  ## Last node defined in the graph
  comp.nodes[^1]

proc rawPtr*(comp: Computation): xla_computation = comp.obj.c

proc name*(comp: Computation): string =
  ## Name of the computation specified when the builder was created + count of number of ops.
  $xla_computation_name(comp.obj.c)

proc paramNames*(comp: Computation): seq[string] =
  ## Names of the parameters which have been defined.
  map(comp.params, x => x.name)

proc `$`*(comp: Computation): string =
  ## Dumps out the name, parameters and info for each node added to the graph.
  let params = comp.paramNames.join(", ")
  result.add &"Computation::{comp.name}({params})"
  for node in comp.nodes:
    result.add "\n" & $node
    if node.args.len > 0:
      result.add "(" & map(node.args, x => $x.id).join(", ") & ")"


proc parameter*(b: Builder, dtype: DataType, dims: openarray[int] = [], name = ""): Node =
  ## Create a new parameter with the given shape. The parameter index is set automatically
  ## based on number of parameters set by this builder. If the name is blank then uses p<index> format.
  let index = b.params.len
  let name = if name == "": "p" & $index else: name
  withDims(dptr, dims):
    let param = parameter(b.obj.c, index, dtype.cint, dims.len.cint, dptr, name.cstring)
    result = b.wrap(param, tParam, info=name)
    result.name = name
    result.paramId = index
    b.params.add result

proc makeTuple*(b: Builder, args: varargs[Node]): Node =
  ## Creates a new tuple from a list of ops.
  if args.len == 0:
    return b.wrap(op_tuple(b.obj.c, nil, 0), tTuple)  
  var ops = cast[ptr xla_op](alloc(args.len*sizeOf(xla_op)))
  for i, arg in args:
    ptrOffset(ops, i)[] = arg.op.c
  result = b.wrap(op_tuple(b.obj.c, ops, csize_t(args.len)), tTuple, args)
  dealloc(ops)

proc iota*(b: Builder, dtype: DataType, length: int): Node =
  ## One dimensional vector of length with values starting from zero.
  let op = b.obj.c.op_iota1(cint(dtype), csize_t(length))
  result = b.wrap(op, tIota, info = &"[{length}]")

proc iota*(b: Builder, dtype: DataType, dims: openarray[int], axis: int): Node =
  ## Creates an array that has specified shape and holds values starting at zero and incrementing by one along 
  ## the specified axis
  withDims(dptr, dims):
    let op = b.obj.c.op_iota(cint(dtype), csize_t(dims.len), dptr, axis)
    result = b.wrap(op, tIota, info = $dims & $axis)

proc errorNode*(b: Builder, message: string): Node =
  ## Node used to record an error e.g. due to invalid input types or shapes.
  b.wrap(b.obj.c.op_invalid_argument_error(message), tError, info="invalid argument")

# forward defs
proc broadcast*(a: Node, dims: openarray[int]): Node
proc convert*(a: Node, dtype: DataType): Node

proc constant*(b: Builder, lit: Literal): Node =
  ## Create new constant from the given literal
  b.wrap(constant_literal(b.obj.c, lit.rawPtr), tLiteral)

proc constant*[T: ElemType](b: Builder, t: Tensor[T]): Node =
  ## Create new constant from the given tensor
  b.constant(t.toLiteral)

# constant builder
macro makeConstant(typ: untyped, ctyp: static string): untyped =
  let const_r0 = ident("constant_r0_" & ctyp)
  let const_r1 = ident("constant_r1_" & ctyp)
  let (b, value) = (ident "b", ident "value")

  result = quote do:
    proc constant*(`b`: Builder, `value`: `typ`): Node =
      ## Create a new scalar constant from the given value.
      `b`.wrap(`const_r0`(`b`.obj.c, `value`), tConst, [], $`value`)

    proc constant*(`b`: Builder, `value`: openarray[`typ`]): Node =
      ## Create a new vector constant from the given value.
      let xop = `const_r1`(`b`.obj.c, `value`[0].unsafeAddr, csize_t(`value`.len))
      `b`.wrap(xop, tConst, [], $`value`)

makeConstant(int32, "int32_t")
makeConstant(int64, "int64_t")
makeConstant(float32, "float")
makeConstant(float64, "double")

proc constant*(b: Builder, value: int): Node =
  ## Create a new int64 scalar constant
  b.constant value.int64

proc constant*(b: Builder, value: openarray[int]): Node =
  ## Create a new int64 vector constant
  b.constant map(value, x => x.int64)

proc constant*[T: float|int](b: Builder, value: T, dtype: Datatype): Node =
  ## Create a new scalar constant with the given type.
  case dtype
  of I32:
    b.constant(value.int32)
  of I64:
    b.constant(value.int64)
  of F32:
    b.constant(value.float32)
  of F64:
    b.constant(value.float64)
  else:
    when T is float:
      convert(b.constant(value.float64), dtype)
    else:
      convert(b.constant(value.int64), dtype)

template `^`*(b: Builder, value: untyped): Node =
  ## Shorthand to generate a new constant node
  b.constant(value)

macro namedConstant(symbol, call, name: untyped, docs: static string) =
  let (b, dtype, dims) = (ident "b", ident "dtype", ident "dims")
  var docComment = newNimNode(nnkCommentStmt)
  docComment.strVal = docs

  result = quote do:
    proc `symbol`*(`b`: Builder, `dtype` = F32, `dims`: openarray[int] = []): Node =
      `docComment`
      let node = `b`.wrap(`call`(`b`.obj.c, cint(`dtype`)), tConst, [], `name`)
      if `dims`.len > 0: broadcast(node, `dims`) else: node

namedConstant(zero, op_zero, "zero",
  "Create a node with the zero value for the given datatype. This is broadcast to dims if provided.")
namedConstant(one, op_one, "one",
  "Create a node with the unit value for the given datatype. This is broadcast to dims if provided.")
namedConstant(minValue, op_min_value, "min_value",
  "Create a node with the minimum value for the given datatype. i.e. -Inf for floating point types." & 
  "This is broadcast to dims if provided.")
namedConstant(maxValue, op_max_value, "max_value",
  "Create a node with the maximum value for the given datatype. i.e. +Inf for floating point types." & 
  "This is broadcast to dims if provided.")

# operations
macro binop(name, opname, typ: untyped, docs: static string): untyped =
  let (a, b) = (ident "a", ident "b")
  var docComment = newNimNode(nnkCommentStmt)
  docComment.strVal = docs
  quote do:
    proc `name`*(`a`, `b`: Node): Node =
      `docComment`
      `a`.builder.wrap(`opname`(`a`.op.c, `b`.op.c), `typ`, [`a`, `b`])

binop(`+`, op_add, tAdd, "Elementwise add")
binop(`-`, op_sub, tSub, "Elementwise subtract")
binop(`*`, op_mul, tMul, "Elementwise multiply")
binop(`/`, op_div, tDiv, "Elementwise divide")
binop(rem, op_rem, tRem, "Elementwise remainder")
binop(max, op_max, tMax, "Elementwise maximum of 2 arrays")
binop(min, op_min, tMin, "Elementwise minimum of 2 arrays")
binop(pow, op_pow, tPow, "Elementwise a raised to power b")
binop(dot, op_dot, tDot, "Vector or matrix doc product per [dot](https://www.tensorflow.org/xla/operation_semantics#dot)")
binop(logicalAnd, op_and, tAnd, "Elementwise logical and between two Bool arrays")
binop(logicalOr, op_or, tOr, "Elementwise logical or between two Bool arrays")
binop(`==`, op_eq, tEq, "Elementwise equal. Returns a Bool array.")
binop(`!=`, op_ne, tNe, "Elementwise not equal. Returns a Bool array.")
binop(`>=`, op_ge, tGe, "Elementwise greater or equal. Returns a Bool array.")
binop(`>`, op_gt, tGt, "Elementwise greater than. Returns a Bool array.")
binop(`<=`, op_le, tLe, "Elementwise less than or equal. Returns a Bool array.")
binop(`<`, op_lt, tLt, "Elementwise less than. Returns a Bool array.")

macro unary(name, opname, typ: untyped, docs: static string): untyped =
  let a = ident "a"
  var docComment = newNimNode(nnkCommentStmt)
  docComment.strVal = docs
  quote do:
    proc `name`*(`a`: Node): Node =
      `docComment`
      `a`.builder.wrap(`opname`(`a`.op.c), `typ`, [`a`])

unary(`!`, op_not, tNot, "Elementwise logical not.")
unary(`-`, op_neg, tNeg, "Elementwise arithmetic negation")
unary(abs, op_abs, tAbs, "Elementwise absolute value")
unary(exp, op_exp, tExp, "Elementwise natural exponential")
unary(floor, op_floor, tFloor, "Elementwise floor rounding")
unary(ceil, op_ceil, tCeil, "Elementwise ceil rounding")
unary(round, op_round, tRound, "Elementwise nearest rounding")
unary(log, op_log, tLog, "Elementwise natural log")
unary(log1p, op_log1p, tLog1p, "Elementwise log(1 + a)")
unary(sigmoid, op_logistic, tSigmoid, "Elementwise 1/(1 + exp(-a))")
unary(sign, op_sign, tSign, "Elementwise sign. Returns -1, 0, +1 or Nan" )
unary(cos, op_cos, tCos, "Elementwise cosine")
unary(sin, op_sin, tSin, "Elementwise sine")
unary(tanh, op_tanh, tTanh, "Elementwise hyperbolic tangent")
unary(sqrt, op_sqrt, tSqrt, "Elementwise square root")
unary(rsqrt, op_rsqrt, tRsqrt, "Elementwise 1/sqrt(a)")
unary(isFinite, op_is_finite, tIsFinite, "Elementwise is not Nan or +=Inf for each. Returns a Bool array.")
unary(copy, op_copy, tCopy, "Returns a copy of the input.")
unary(zerosLike, op_zeros_like, tZerosLike, "Creates a new zero value with element type and shape from input.")

proc normalize(index, rank: int): int =
  if index < 0: rank+index else: index

proc `[]`*(a: Node, index: int): Node =
  ## Return the element from the input tuple at index.
  result = a.builder.wrap(op_get_tuple_element(a.op.c, index), tTupleElement, [a], $index)
  result.index = index

proc convert*(a: Node, dtype: DataType): Node =
  ## Convert type of elements to dtype.
  a.builder.wrap(op_convert_element_type(a.op.c, cint(dtype)), tConvert, [a], $dtype)

proc reshape*(a: Node, dims: varargs[int]): Node =
  ## Reshape the input node to dims. Total number of elements is unchanged.
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  let dims2 = reshapeDims(prod(a.dims), dims)
  withDims(dptr, dims2):
   result = a.builder.wrap(op_reshape(a.op.c, csize_t(dims2.len), dptr), tReshape, [a], $dims2)
   result.indices = dims2

proc flatten*(a: Node, startDim = 0, endDim = -1): Node =
  ## Reshapes input such that any dimensions between startDim and endDim are flattened. e.g. flatten(a, 1) 
  ## returns a 2d array keeping the first dimension and collapsing all the remainder into the second.
  let first = normalize(startDim, a.rank)
  let last = normalize(endDim, a.rank)
  let d = a.dims
  var dims: seq[int]
  if first > 0:
    dims.add d[0 ..< first]
  if last > first:
    dims.add prod(d[first .. last])
  if last < d.len-1:
    dims.add d[last+1 .. ^1]
  a.reshape(dims)

proc reverse*(a: Node, axes: varargs[int]): Node =
  ## Reverse the elements in the input along the given axes.
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  let axes2 = map(axes, x => normalize(x, a.rank))
  withDims(dptr, axes2):
   result = a.builder.wrap(op_reverse(a.op.c, csize_t(axes2.len), dptr), tReverse, [a], $axes2)
   result.indices = axes2

proc broadcast*(a: Node, dims: openarray[int]): Node =
  ## Add new leading dimensions to the input node per
  ## [Broadcast](https://www.tensorflow.org/xla/operation_semantics#broadcast)
  withDims(dptr, dims):
    result = a.builder.wrap(op_broadcast(a.op.c, csize_t(dims.len), dptr), tBroadcast, [a], $dims)
    result.indices = @dims

proc broadcastInDim*(a: Node, outSize, bcastDims: openarray[int]): Node =
  ## Expand dims at each index in bcastDimes from 1 to the corresponding value in outSize as 
  ## per [BroadcastInDim](https://www.tensorflow.org/xla/operation_semantics#broadcastindim)
  withDims(dptr1, outSize):
    withDims(dptr2, bcastDims):
      let op = op_broadcast_in_dim(a.op.c, csize_t(outSize.len), dptr1, csize_t(bcastDims.len), dptr2)
      result = a.builder.wrap(op, tBroadcastInDim, [a], &"(outSize:{outSize} bcastDims:{bcastDims})")
      result.outSize = @outSize
      result.bcastDims = @bcastDims

# collapse given dimensions
proc collapse*(a: Node, dims: openarray[int]): Node =
  ## Collapse the given dimensions into a single dimension as per 
  ## [Collapse](https://www.tensorflow.org/xla/operation_semantics#collapse).
  ## dims should be an in-order consecutive subset of the input dims.
  withDims(dptr, dims):
    result = a.builder.wrap(op_collapse(a.op.c, csize_t(dims.len), dptr), tCollapse, [a], $dims)
    result.indices = @dims

proc transpose*(a: Node, axes: varargs[int]): Node =
  ## Permute the the given axes. If no axes are given then will swap the last 2 axes.
  ## Axes indices may be negative - in this case they will be relative to the number of dimensions.
  var axes2: seq[int]
  if axes.len == 0 and a.rank >= 2:
    axes2 = @[a.rank-1, a.rank-2]
  else:
    axes2 = map(axes, x => normalize(x, a.rank))
  withDims(dptr, axes2):
    result = a.builder.wrap(op_transpose(a.op.c, csize_t(axes2.len), dptr), tTranspose, [a], $axes2)
    result.indices = axes2

proc narrow*(a: Node, dim, start, stop: int, stride = 1): Node =
  ## Returns the data narrowed such that dimension dim ranges from start..stop-1 with step of stride.
  ## As per [Slice](https://www.tensorflow.org/xla/operation_semantics#slice)
  let op = op_slice_in_dim(a.op.c, start, stop, stride, dim)
  result = a.builder.wrap(op, tNarrow, [a], &"(dim:{dim} start:{start} stop:{stop} stride:{stride})")
  result.indices = @[dim, start, stop]

proc relu*(a: Node): Node =
  ## Rectified linear unit activation function: max(0, a)
  result = max(a.builder.zero(a.dtype), a)
  result.kind = tRelu
  result.args = @[a]

proc select*(a, onTrue, onFalse: Node): Node =
  ## Select values from onTrue where a is true else from onFalse.
  a.builder.wrap(op_select(a.op.c, onTrue.op.c, onFalse.op.c), tSelect, [a, onTrue, onFalse])

proc clamp*(a, min, max: Node): Node =
  ## Clamp values in a to be between min and max.
  a.builder.wrap(op_clamp(min.op.c, a.op.c, max.op.c), tClamp, [a, min, max])

proc rngUniform*(min, max: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a uniform random distribution with values from min to max and 
  ## given dimensions. Inputs must have the same data type. This is used as the element type for the output.
  withDims(dptr, dims):
    let op = op_rng_uniform(min.op.c, max.op.c, cint(min.dtype), cint(dims.len), dptr)
    result = min.builder.wrap(op, tRngUniform, [min, max], $dims)
    result.indices = @dims

proc rngNormal*(mean, stddev: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a normal random distribution described by mean, std deviation, 
  ## data type and dimensions. Inputs must have the same data type. This is used the as element type for the output.
  withDims(dptr, dims):
    let op = op_rng_normal(mean.op.c, stddev.op.c, cint(mean.dtype), cint(dims.len), dptr)
    result = mean.builder.wrap(op, tRngNormal, [mean, stddev], $dims)
    result.indices = @dims

proc concat*(a: Node, nodes: openarray[Node], axis: int): Node =
  ## Concatenate the given nodes with a along the given axis.
  let axis = normalize(axis, a.rank)
  var args = map(nodes, x => x.op.c)
  let op = op_concat_in_dim(a.op.c, args[0].addr, csize_t(args.len), axis)
  result = a.builder.wrap(op, tConcat, nodes, &"axis={axis}")
  result.index = axis

proc getPadding(padding: openarray[Padding], kernelSize: openarray[int]): (seq[int64], seq[int64]) =
  var padLo, padHi: seq[int64]
  for i, p in padding:
    if p.same:
      let w = kernelSize[i].int64
      padLo.add (w - 1) div 2
      padHi.add w div 2
    else:
      padLo.add p.lo.int64
      padHi.add p.hi.int64
  return (padLo, padHi)

proc convolution*(a, kernel: Node, inputDims, outputDims, kernelDims: openarray[int],
                  strides: openarray[int] = [], padding: openarray[Padding] = [],
                  dilation, inputDilation: openarray[int] = [], groups, batchGroups = 1): Node =
  ## General n dimensional convolution call. See conv1d, conv2d and conv3d for simplified version.
  ##
  ## inputDims, outputDims and kernelDims provide the layout for the dimensions of each tensor
  ## - dims[0] = batch / kernel output dimension
  ## - dims[1] = channel / kernel input dimension
  ## - dims[2..] = spatial dimensions
  ## If set then strides, padding and dilation should have same number of entries as spatial dimensions.
  ##
  ## See [ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution).
  if inputDims.len != outputDims.len or inputDims.len != kernelDims.len:
    raiseError("convolution: number input/output/kernel dimensions should be equal", a)
  if inputDims.len < 3:
    raiseError("convolution: number input/output/kernel dimensions should be at least 3", a)
  let nd = inputDims.len - 2
  var idims = map(inputDims, x => x.int64)
  var odims = map(outputDims, x => x.int64)
  var kdims = map(kernelDims, x => x.int64)
  var stride = i64seq(strides, nd, 1, "stride")
  var inDilation = i64seq(inputDilation, nd, 1, "input dilation")
  var kDilation = i64seq(dilation, nd, 1, "dilation")
  if padding.len != 0 and padding.len != nd:
    raiseError(&"convolution: number of padding values should be {nd} - got {padding.len}")
  let kernelSize = collect:
    for i in 0 ..< padding.len:
      int((kernel.dims[kdims[i+2]] - 1) * kDilation[i] + 1)
  var (padLo, padHi) = getPadding(padding, kernelSize)
  let op = op_conv(a.op.c, kernel.op.c, csize_t(nd), idims[0].addr, odims[0].addr, kdims[0].addr,
                   stride[0].addr, inDilation[0].addr, kDilation[0].addr, csize_t(padding.len), 
                   padLo[0].addr, padHi[0].addr, groups, batchGroups)
  result = a.builder.wrap(op, tConv, [a, kernel], &"strides={strides},padding={padding}")
  if dilation.len > 0:
    result.info.add &",dilation={dilation}"
  if groups != 1:
    result.info.add &",groups={groups}"
  result.idims = @inputDims
  result.odims = @outputDims
  result.kdims = @kernelDims
  result.strides = @strides
  result.padding = @padding
  result.dilation = @dilation
  result.groups = groups

proc getDims*(ndims: int, channelsFirst: bool): seq[int] =
  ## Get dimension index numbers for given layout.
  if channelsFirst:
    @[0, 1] & toSeq(2 .. ndims+1)
  else:
    @[0, ndims+1] & toSeq(1 .. ndims)

proc conv1d*(a, kernel: Node, strides=1, padding=pad(0), dilation=1, groups=1, channelsFirst=false): Node =
  ## One dimensional convolution with optional low and high padding and either strides or dilation.
  ##
  ## Default layout should be
  ## - a:      [N, W, C]
  ## - kernel: [K, W, C]
  ##
  ## If channelsFirst is set then
  ## - a:      [N, C, W]
  ## - kernel: [K, C, W]
  ##
  ## Where N = number of batches, C = input channels, K = output channels and W is the spatial dimension.
  ## If groups is > 1 then performs a grouped convolution. In this case C and K should be divisible by groups.
  if a.rank != 3 or kernel.rank != 3:
    raiseError(&"conv1d: input and kernel rank should be 3 - have {a.rank},{kernel.rank}", a)
  let dims = getDims(1, channelsFirst)
  convolution(a, kernel, dims, dims, dims, [strides], [padding], [dilation], groups=groups)

proc conv2d*(a, kernel: Node, strides: Opt2d = 1, padding: Pad2d = pad(0), dilation: Opt2d = 1, 
             groups=1, channelsFirst=false): Node =
  ## Two dimensional convolution with optional padding and either strides or dilation.
  ##
  ## Default layout should be
  ## - a:      [N, H, W, C]
  ## - kernel: [K, H, W, C]
  ##
  ## If channelsFirst is set then
  ## - a:      [N, C, H, W]
  ## - kernel: [K, C, H, W]
  ##
  ## Where N = number of batches, C = input channels, K = output channels and H, W are spatial dimensions.
  ## If groups is > 1 then performs a grouped convolution. In this case C and K should be divisible by groups.
  if a.rank != 4 or kernel.rank != 4:
    raiseError(&"conv2d: input and kernel rank should be 4 - have {a.rank},{kernel.rank}", a)
  let dims = getDims(2, channelsFirst)
  convolution(a, kernel, dims, dims, dims, strides.seq2, padding.seq2, dilation.seq2, groups=groups)

proc conv3d*(a, kernel: Node, strides: Opt3d = 1, padding: Pad3d = pad(0), dilation: Opt3d = 1, 
             groups=1, channelsFirst=false): Node =
  ## Three dimensional convolution with optional padding and either strides or dilation.
  ##
  ## Default layout should be
  ## - a:      [N, D, H, W, C]
  ## - kernel: [K, D, H, W, C]
  ##
  ## If channelsFirst is set then
  ## - a:      [N, C, D, H, W]
  ## - kernel: [K, C, D, H, W]
  ##
  ## Where N = number of batches, C = input channels, K = output channels and D, H, W are spatial dimensions.
  ## If groups is > 1 then performs a grouped convolution. In this case C and K should be divisible by groups.
  if a.rank != 5 or kernel.rank != 5:
    raiseError(&"conv1d: input and kernel rank should be 5 - have {a.rank},{kernel.rank}", a)
  let dims = getDims(3, channelsFirst)
  convolution(a, kernel, dims, dims, dims, strides.seq3, padding.seq3, dilation.seq3, groups=groups)

proc gather*(a, indices: Node): Node =
  ## Builds a new tensor by taking individual values from the original tensor at the given indices.
  ## The last dimension in indices should have the same size as the tensor rank, i.e. you can
  ## think of indices as a 'list' of indexes which we iterate over each one of which describes a position
  ## in the source array.
  ##
  ## For example:
  ## ```
  ## a = <f32 2 2>[[1 2]    ix = <i32 3 2>[[1 1]
  ##                3 4]]                  [0 1]
  ##                                       [1 0]]
  ## a.gather(ix) = <f32 3>[4 2 3]
  ## ```
  ##
  ## This is a simplified version of the [Gather](https://www.tensorflow.org/xla/operation_semantics#gather) op.
  let sliceSizes = repeat(1, a.rank)
  let axes = toSeq(0 ..< a.rank)
  let ndims = csize_t(a.rank)
  withDims(dptr1, axes):
    withDims(dptr2, sliceSizes):
      let op = op_gather(a.op.c, indices.op.c,
                        nil, 0,                     # offset_dims
                        dptr1, ndims,               # collapsed_slice_dims
                        dptr1, ndims,               # start_index_map
                        indices.rank-1,             # index_vector_dim
                        dptr2, ndims)               # slice_sizes
      result = a.builder.wrap(op, tGather, [a, indices])

proc scatter*(a, indices, b: Node, comp: Computation): Node =
  ## Get the values of input array a at the specified indices and updated with values
  ## in b using comp. Indices should have shape [n, a.rank] - i.e. each row is an element
  ## to update and the columns indicate the location in the target vector (which can be repeated.)
  ##
  ## This is the opposite of gather. It is a simplified version of the
  ## [Scatter](https://www.tensorflow.org/xla/operation_semantics#scatter) for details.
  var ixdims = indices.dims
  if ixdims.len != 2 or ixdims[1] != a.rank:
    raiseError("Index dimensions for scatter should be [n, a.rank] - got " & $ixdims, a)
  # reshape indices to [..., x, y] and updates to [..., x]
  let rankDiff = a.rank - 1
  if rankDiff > 0:
    ixdims = repeat(1, rankDiff) & ixdims
  let indices = indices.reshape(ixdims)
  let b = b.reshape(ixdims[0 .. ^2])
  let axes = toSeq(0 ..< a.rank)
  let ndims = csize_t(a.rank)
  withDims(dptr, axes):
    let op = op_scatter(a.op.c, indices.op.c, b.op.c, comp.obj.c,
                      a.rank,                  # index_vector_dim
                      nil, 0,                  # update_window_dims
                      dptr, ndims,             # inserted_window_dims
                      dptr, ndims)             # scatter_dims_to_operand_dims
    result = a.builder.wrap(op, tScatter, [a, indices, b])

proc addAt*(a, indices, b: Node): Node =
  ## Adds values from array b to array a at the given indices.
  ##
  ## For example:
  ## ```
  ## a = <f32 2 3>[[1 2 3]    ix = <i64 3 2>[[0 0]   b = <f32 3>[1 2 3]
  ##               [4 5 6]]                  [1 2]
  ##                                         [0 0]]
  ## a.addAt(ix, b) = <f32 2 3>[[5 2 3]
  ##                             4 5 8]]
  ## ```
  ## This is implemented using the scatter op.
  let b2 = newBuilder("addAt")
  let sum = b2.build(b2.parameter(a.dtype) + b2.parameter(a.dtype))
  a.scatter(indices, b, sum)


proc reduceDims(shape: seq[int], dims: openarray[int]): seq[int] =
  ## Normalized dimensions for reduction op
  if dims.len == 0:
    toSeq(0 ..< shape.len)
  else:
    map(dims, x => normalize(x, shape.len))

proc reduce*(a, initValue: Node, comp: Computation, dims: openarray[int] = [], 
            nodeType = tReduce, keepDims = false): Node =
  ## Apply reduction across one or more dimensions.  i.e. comp is applied repeatedly with a pair of elements
  ## from the a input node. initValue defines the initial 'zero' value for the reduction.
  ## If no dims given then the reduction is applied across all of the input dimensions to reduce to a scalar.
  ## If the dimension index is negative then it is relative to the number of dimensions.
  ## If keepDims is set then the summed dimensions are kept with a size of 1, else they are removed
  ## and the numbe of dimensions in the result is reduced.
  var shape = a.dims
  var dims2 = reduceDims(shape, dims)
  withDims(dptr, dims2):
    let op = op_reduce(a.op.c, initValue.op.c, comp.obj.c, dptr, csize_t(dims2.len))
    var info = $dims2
    if keepDims: info.add ":keepDims" 
    result = a.builder.wrap(op, nodeType, [a, initValue], info)
    result.indices = dims2
    if keepDims and dims2.len > 0:
      for d in dims2: shape[d] = 1
      result = result.reshape(shape)

proc sum*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to sum of elements across one or more dimensions in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = newBuilder("reduce")
  let sum = b.build(b.parameter(a.dtype) + b.parameter(a.dtype))
  reduce(a, a.builder.zero(a.dtype), sum, dims, tReduceSum, keepDims)

proc sum*(a: Node, axis: int, keepDims = false): Node =
  ## Reduce to sum of elements across the given axis in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  sum(a, [axis], keepDims)

proc mean*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to mean of elements across one or more dimensions in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  var scale = 1
  for i in reduceDims(a.dims, dims):
    scale *= a.dims[i]
  a.sum(dims, keepDims) / a.builder.constant(scale, a.dtype)

proc mean*(a: Node, axis: int, keepDims = false): Node =
  ## Reduce to mean of elements across the given axis in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  mean(a, [axis], keepDims)

proc min*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to minimum value of elements across one or more dimensions in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = newBuilder("reduce")
  let sum = b.build(min(b.parameter(a.dtype), b.parameter(a.dtype)))
  reduce(a, a.builder.maxValue(a.dtype), sum, dims, tReduceMin, keepDims)

proc min*(a: Node, axis: int, keepDims = false): Node =
  ## Reduce to minimum value of elements across the given axis in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  min(a, [axis], keepDims)

proc max*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to maximum value of elements across one or more dimensions in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = newBuilder("reduce")
  let sum = b.build(max(b.parameter(a.dtype), b.parameter(a.dtype)))
  reduce(a, a.builder.minValue(a.dtype), sum, dims, tReduceMax, keepDims)

proc max*(a: Node, axis: int, keepDims = false): Node =
  ## Reduce to maximum value of elements across the given axis in the input. 
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  max(a, [axis], keepDims)

macro argMinMax(procName, compare, initVal, opType: untyped): untyped =
  let (a, axis, keepDims, ixType) = (ident "a", ident "axis", ident "keepDims", ident "ixType")
  quote do:
    proc `procName`*(`a`: Node, `axis`: int, `keepDims` = false, `ixType` = I64): Node =
      ## Get the indices of the minimum or maxiumum values along the given axis for argmin and argmax respectively.
      ## By default the shape of the result will be as per the input with this axis removed.
      ## If keepDims is set the axis for the reduction is kept in the output with size of 1.
      ## If a negative axis is given then this is taken relative to the number of dimensions of the input.
      let b = `a`.builder
      var shape = `a`.dims
      let dtype = `a`.dtype
      let b2 = newBuilder("reduce")
      let (v0, i0) = (b2.parameter(dtype), b2.parameter(`ixType`))
      let (v1, i1) = (b2.parameter(dtype), b2.parameter(`ixType`))
      let comp = b2.build b2.makeTuple(
        # compare values
        select(`compare`(v0, v1), v0, v1),
        # compare indices
        select(v0 == v1, min(i0, i1), select(`compare`(v0, v1), i0, i1))
      )
      let initValue = b.`initVal`(dtype)
      let initIndex = b.zero(`ixType`)
      let axis = normalize(`axis`, shape.len)
      let indexes = b.iota(`ixType`, shape, axis)
      var dims = @[`axis`]
      withDims(dptr, dims):
        let op = op_reduce2(b.obj.c, `a`.op.c, initValue.op.c, indexes.op.c, initIndex.op.c, 
                            comp.obj.c, dptr, csize_t(dims.len))
        var info = $dims
        if `keepDims`: info.add ":keepDims" 
        # calc both min/max and argmin/argmax, but just use the latter
        xla_op_free(op_get_tuple_element(op, 0))
        result = b.wrap(op_get_tuple_element(op, 1), `opType`, [], info)
        result.indices = dims
        if `keepDims`:
          shape[axis] = 1
          result = result.reshape(shape)

argMinMax(argMax, `>=`, minValue, tArgmax)
argMinMax(argMin, `<=`, maxValue, tArgmin)


proc poolDims(val: openarray[int], channelsFirst: bool): seq[int] =
  if channelsFirst:
    @[1, 1] & @val
  else:
    @[1] & @val & @[1]

proc poolPadding(val: openarray[Padding], channelsFirst: bool): seq[Padding] =
  for v in val:
    if v != pad(0):
      return if channelsFirst:
        @[pad(0), pad(0)] & @val
      else:
        @[pad(0)] & @val & @[pad(0)]
  return @[]

proc reduceWindow*(a, initValue: Node, comp: Computation, windowDims, strides: openarray[int], padding: openarray[Padding] = [],
                   nodeType = tReduceWindow): Node =
  ## Apply reduction to all elements in each window of a sequence of N multi-dimensional arrays.
  ## The mumber of entries in the windowDims and strides array should equal the rank of the input array
  ## (i.e. entries should be 1 for non-spatial dimensions).
  ## If set then number of entries in the padding array should also equal the input rank, where non-spatial
  ## dimensions have padding of 0.
  ##
  ## This can be used to implement pooling layers.
  ## See [ReduceWindow](https://www.tensorflow.org/xla/operation_semantics#reducewindow) for details.
  let rank = a.rank
  if windowDims.len != rank or strides.len != rank:
    raiseError(&"reduceWindow: windowDims and strides length should equal input rank {rank} - have {windowDims}, {strides}", a)
  if padding.len != 0 and padding.len != rank:
    raiseError(&"reduceWindow: padding length should equal input rank {rank} - have {padding}", a)
  var (padLo, padHi) = getPadding(padding, windowDims)
  withDims(dptr1, windowDims):
    withDims(dptr2, strides):
      let (p1, p2) = if padding.len > 0: (padLo[0].addr, padHi[0].addr) else: (nil, nil)
      let op = op_reduce_window(a.op.c, initValue.op.c, comp.obj.c, csize_t(rank), dptr1, dptr2,
                                csize_t(padding.len), p1, p2)
      result = a.builder.wrap(op, nodeType, [a, initValue], &"dims={windowDims},strides={strides},padding={padding}")
      result.wdims = @windowDims
      result.wstride = @strides
      result.wpad = @padding

proc maxPool1d*(a: Node, kernelSize: int, strides = 0, padding = pad(0), channelsFirst=false): Node =
  ## Max pooling over 1 dimensional input array. Stride defaults to kernelSize if left as 0.
  ## channelsFirst setting is as per conv1d.
  if a.rank != 3: raiseError(&"maxPool1d: input rank should be 3 - have {a.rank}", a)
  let strides = if strides == 0: kernelSize else: strides
  let b = newBuilder("reduceWindow")
  let sum = b.build(max(b.parameter(a.dtype), b.parameter(a.dtype)))
  let windowDims = poolDims([kernelSize], channelsFirst)
  let strideDims = poolDims([strides], channelsFirst)
  let padDims = poolPadding([padding], channelsFirst)
  reduceWindow(a, a.builder.minValue(a.dtype), sum, windowDims, strideDims, padDims, nodeType=tMaxPool)

proc maxPool2d*(a: Node, kernelSize: Opt2d, strides: Opt2d = 0, padding: Pad2d = pad(0), channelsFirst=false): Node =
  ## Max pooling over 2 dimensional input array. Stride defaults to kernelSize if left as 0.
  ## channelsFirst setting is as per conv2d.
  if a.rank != 4: raiseError(&"maxPool2d: input rank should be 4 - have {a.rank}", a)
  let strides = if strides == 0: kernelSize else: strides
  let b = newBuilder("reduceWindow")
  let sum = b.build(max(b.parameter(a.dtype), b.parameter(a.dtype)))
  let windowDims = poolDims(kernelSize.seq2, channelsFirst)
  let strideDims = poolDims(strides.seq2, channelsFirst)
  let padDims = poolPadding(padding.seq2, channelsFirst)
  reduceWindow(a, a.builder.minValue(a.dtype), sum, windowDims, strideDims, padDims, nodeType=tMaxPool)

proc maxPool3d*(a: Node, kernelSize: Opt3d, strides: Opt3d = 0, padding: Pad3d = pad(0), channelsFirst=false): Node =
  ## Max pooling over 3 dimensional input array. Stride defaults to kernelSize if left as 0.
  ## channelsFirst setting is as per conv3d.
  if a.rank != 5: raiseError(&"maxPool2d: input rank should be 5 - have {a.rank}", a)
  let strides = if strides == 0: kernelSize else: strides
  let b = newBuilder("reduceWindow")
  let sum = b.build(max(b.parameter(a.dtype), b.parameter(a.dtype)))
  let windowDims = poolDims(kernelSize.seq3, channelsFirst)
  let strideDims = poolDims(strides.seq3, channelsFirst)
  let padDims = poolPadding(padding.seq3, channelsFirst)
  reduceWindow(a, a.builder.minValue(a.dtype), sum, windowDims, strideDims, padDims, nodeType=tMaxPool)


proc selectAndScatter*(a, source: Node, windowDims, strides: openarray[int], padding: openarray[Padding] = []): Node =
  ## Composite operation that first computes ReduceWindow on the operand array to select an element from each
  ## window, and then scatters the source array to the indices of the selected elements to construct an output
  ## array with the same shape as the operand array.
  ##
  ## Used for gradient of the maxPool function.
  ## See [SelectAndScatter](https://www.tensorflow.org/xla/operation_semantics#selectandscatter) for details.
  let rank = a.rank
  if windowDims.len != rank or strides.len != rank:
    raiseError(&"selectAndScatter windowDims and strides length should equal input rank {rank} - have {windowDims}, {strides}", a)
  if padding.len != 0 and padding.len != rank:
    raiseError(&"selectAndScatter: padding length should equal input rank {rank} - have {padding}", a)
  var (padLo, padHi) = getPadding(padding, windowDims)
  let b1 = newBuilder("select")
  let selectComp = b1.build(b1.parameter(a.dtype) >= b1.parameter(a.dtype))
  let b2 = newBuilder("scatter")
  let scatterComp = b2.build(b2.parameter(a.dtype) + b2.parameter(a.dtype))
  let init = a.builder.zero(a.dtype)
  withDims(dptr1, windowDims):
    withDims(dptr2, strides):
      let (p1, p2) = if padding.len > 0: (padLo[0].addr, padHi[0].addr) else: (nil, nil)
      let op = op_select_and_scatter(a.op.c, selectComp.obj.c, csize_t(rank), dptr1, dptr2, 
                         csize_t(padding.len), p1, p2, source.op.c, init.op.c, scatterComp.obj.c)
      result = a.builder.wrap(op, tSelectAndScatter, [a, source])

proc batchNormInference*(a, scale, offset, mean, variance: Node, epsilon: float, axis: int): Node =
  ## Implements batch normalization in inference mode.
  ## axis should be the axis of the feature dimension - e.g. 3 or -1 for images in [N,H,W,C] format.
  ## See [BatchNormInference](https://www.tensorflow.org/xla/operation_semantics#batchnorminference).
  let axis = normalize(axis, a.rank)
  let op = op_batch_norm_inference(a.op.c, scale.op.c, offset.op.c, mean.op.c, variance.op.c, epsilon, axis)
  result = a.builder.wrap(op, tBatchNormInference, [a, scale, offset, mean, variance], &"axis={axis}")
  result.epsilon = epsilon
  result.axis = axis

proc batchNormTraining*(a, scale, offset: Node, epsilon: float, axis: int): Node =
  ## Implements batch normalization in training mode. Returns a tuple of (output, batch_mean, batch_var)
  ## See [BatchNormTraining](https://www.tensorflow.org/xla/operation_semantics#batchnormtraining).
  let axis = normalize(axis, a.rank)
  let op = op_batch_norm_training(a.op.c, scale.op.c, offset.op.c, epsilon, axis)
  result = a.builder.wrap(op, tBatchNormTraining, [a, scale, offset], &"axis={axis}")
  result.epsilon = epsilon
  result.axis = axis

proc batchNormGrad*(a, scale, mean, variance, gradOutput: Node, epsilon: float, axis: int): Node =
  ## Calculates gradient of batch norm. Returns a tuple of (grad_a, grad_scale, grad_offset)
  ## See [BatchNormGrad](https://www.tensorflow.org/xla/operation_semantics#batchnormgrad).
  let axis = normalize(axis, a.rank)
  let op = op_batch_norm_grad(a.op.c, scale.op.c, mean.op.c, variance.op.c, gradOutput.op.c, epsilon, axis)
  result = a.builder.wrap(op, tBatchNormGrad, info = &"axis={axis}")
  result.epsilon = epsilon
  result.axis = axis

proc unbcast(b: Builder, v: Node, xd, yd: openarray[int]): Node =
  ## If inputs to a 2 op node are different shapes and broadcasting was applied then we need to 
  ## take account of this and cast the gradient back to the original shape when propagating it backward
  var axes: seq[int]
  let ndim = max(xd.len, yd.len)
  let xShape = repeat(1, ndim-xd.len) & @xd
  let yShape = repeat(1, ndim-yd.len) & @yd
  for i, (dx, dy) in zip(xShape, yShape):
    if dx == 1 and dy > 1: axes.add i
  if axes.len > 0: v.sum(axes) else: v

proc unreduce(n, x, v: Node): Node =
  ## Inverse of reduceSum operation, broadcast back to original shape
  var shape = x.dims
  for d in n.indices: shape[d] = 1
  let dims = toSeq(0 ..< shape.len)
  return v.reshape(shape).broadcastInDim(x.dims, dims)

proc dotgrad(n, x, y, v: Node): seq[Node] =
  ## Gradient for dot product node
  case n.rank
  of 0:  # vector product: [n] x [n] => []
    return @[ v*y, v*x ]
  of 1:  # matrix vector product: [m, k] x [k] => [m]
    return @[ dot(v.reshape(-1, 1), y.reshape(1, -1)), dot(v, x) ]
  of 2: # matrix product: [m, k] x [k, n] => [m, n]
    return @[ dot(v, y.transpose), dot(x.transpose, v) ]
  else:
    raiseError("dot gradient not implemented for > 2 dimensions", n)

proc getConvAxis(n, x, kernel: Node, axis: int): (int, int, int, int, int, Padding) =
  let isize = x.dims[n.idims[axis+2]]
  let osize = n.dims[n.odims[axis+2]]
  let ksize = kernel.dims[n.kdims[axis+2]]
  let stride = if n.strides.len > 0: n.strides[axis] else: 1
  let dilation = if n.dilation.len > 0: n.dilation[axis] else: 1
  let pads = if n.padding.len > 0: n.padding[axis] else: pad(0)
  (isize, osize, ksize, stride, dilation, pads)

proc convInputGrad(n, x, kernel, v: Node): Node =
  ## Gradient of convolution with respect to input
  let revKernel = kernel.reverse(n.kdims[2 .. ^1])
  let kernelDims = @[n.kdims[1], n.kdims[0]] & n.kdims[2 .. ^1]
  var paddings: seq[Padding]
  for axis in 0 ..< n.rank-2:
    let (isize, osize, ksize, stride, dilation, pads) = getConvAxis(n, x, kernel, axis)
    let ksize2 = (ksize-1) * dilation + 1
    let inputStart = (ksize2 - 1) div 2 - pads.lo
    let inputEnd = isize - ksize2 div 2 + pads.hi
    assert inputEnd - inputStart + (stride-1) div stride == osize
    let outputStart = -inputStart - ((ksize2 - 1) div 2)
    assert outputStart <= 0
    let outputEnd = isize - inputStart + ksize2 div 2 - (osize-1) * (stride - 1)
    assert outputEnd >= osize
    paddings.add pad(-outputStart, outputEnd - osize)
  convolution(v, revKernel, n.idims, n.odims, kernelDims, padding=paddings,
              dilation=n.dilation, inputDilation=n.strides)

proc expectedOutputSize(isize, ksize, dilation, stride: int, pads: Padding): int =
  let ksize = (ksize-1)*dilation + 1
  let istart = (ksize-1) div 2 - pads.lo
  let iend = isize - ksize div 2 + pads.hi
  (iend - istart + stride - 1) div stride

proc convKernelGrad(n, x, kernel, v: Node): Node =
  ## Gradient of convolution with respect to kernel
  let inputDims = @[n.idims[1], n.idims[0]] & n.idims[2 .. ^1]
  let outputDims = @[n.kdims[1], n.kdims[0]] & n.kdims[2 .. ^1]
  let kernelDims = @[n.odims[1], n.odims[0]] & n.odims[2 .. ^1]
  var paddings = n.padding
  for axis in 0 ..< n.rank-2:
    let (isize, osize, ksize, stride, dilation, pads) = getConvAxis(n, x, kernel, axis)
    assert osize == expectedOutputSize(isize, ksize, dilation, stride, pads)
    let reverseSize = expectedOutputSize(isize, osize, stride, dilation, pads)
    paddings[axis].hi += (ksize - reverseSize) * dilation
  convolution(x, v, inputDims, outputDims, kernelDims, strides=n.dilation,
              padding=paddings, dilation=n.strides)

proc localGrad(b: Builder, n, v: Node): seq[Node] =
  ## Returns the gradients at each of the inputs to node as a function of the value accumulated so far.
  ## Will raise a BuilderError exception if the node type is not supported.
  var x, y: Node
  if n.len >= 1: x = n.args[0]
  if n.len >= 2: y = n.args[1]
  case n.kind
  of tAdd:
    return @[ 
      b.unbcast(v, x.dims, y.dims),
      b.unbcast(v, y.dims, x.dims)
    ]
  of tSub:
    return @[ 
      b.unbcast(v, x.dims, y.dims),
      b.unbcast(-v, y.dims, x.dims)
    ]
  of tMul:
    return @[ 
      b.unbcast(v*y, x.dims, y.dims),
      b.unbcast(v*x, y.dims, x.dims)
    ]
  of tDiv:
    return @[ 
      b.unbcast(v/y, x.dims, y.dims),
      b.unbcast(-v*x/(y*y), y.dims, x.dims)
    ]
  of tDot:
    return n.dotGrad(x, y, v)
  of tConv:
    return @[ n.convInputGrad(x, y, v), n.convKernelGrad(x, y, v) ]
  of tMaxPool:
    return @[ selectAndScatter(x, v, n.wdims, n.wstride, n.wpad) ]
  of tNeg:
    return @[ -v ]
  of tExp:
    return @[ v*n ]
  of tLog:
    return @[ v / x ]
  of tLog1p:
    return @[ v / (b.one(x.dtype) + x) ]
  of tSqrt:
    return @[ v * b.constant(0.5, n.dtype) * x ]
  of tRsqrt:
    return @[ v * b.constant(-0.5, n.dtype) * x ]
  of tTanh:
    return @[ v * (b.one(x.dtype) - n * n) ]
  of tAbs:
    return @[ v * sign(x) ]
  of tSigmoid:
    return @[ v * n * (b.one(x.dtype) - n) ]
  of tRelu:
    let zero = b.zero(x.dtype)
    return @[ select(x >= zero, v, zero) ]
  of tReduceSum:
    return @[ unreduce(n, x, v) ]
  of tReshape:
    return @[ v.reshape(x.dims) ]
  of tTranspose:
    return @[ v.transpose(x.indices) ]
  of tConvert:
    return @[ v.convert(x.dtype) ]
  of tGather:
    let indices = y.reshape(-1, x.rank)
    return @[ b.zero(x.dtype, x.dims).addAt(indices, v) ]
  of tSelect:
    let zero = b.zero(v.dtype, v.dims)
    return @[ Node(kind: tNone), select(x, v, zero), select(x, zero, v) ]
  of tBroadcast:
    if x.rank == 0: return @[ v.sum() ]
  of tBatchNormTraining:
    let grads = batchNormGrad(x, y, n[1], n[2], v, n.epsilon, n.axis)
    return @[ grads[0], grads[1], grads[2] ]
  else:
    raiseError("Node type not supported for autograd", n)

proc calcGrads(b: Builder, node, pathValue: Node, inputs: openarray[string], 
                grads: var openarray[Node], dict: var Table[uint64, Node]) =
  ## Recursively accumulate gradients from node where pathValue is prior value to this point
  let node = if node.kind == tTupleElement:
    node.args[0]
  else:
    node
  debug "calcGrads: ", node
  for i, grad in b.localGrad(node, pathValue):
    if grad.kind == tNone: continue
    let input = node.args[i]
    let id = input.uid
    dict[id] = if dict.hasKey(id):
      dict[id] + grad
    else:
      grad
    if input.len > 0 and not input.noGrad:
      b.calcGrads(input, grad, inputs, grads, dict)
    elif input.kind == tParam:
      let n = inputs.find(input.name)
      if n >= 0:
        grads[n] = dict[id]

proc reshapeAs(n: Node, b: Builder, name: string): Node =
  ## Returns n reshaped to match the parameter with the given name.
  for p in b.params:
    if p.name == name:
      if n.dims == p.dims:
        return n
      else:
        return n.reshape(p.dims)
  raiseError("gradient cannot be calculated for " & name & " parameter not found", n)

proc gradient*(b: Builder, output: Node, inputs: openarray[string]): seq[Node] =
  ## Generate the graph to calculate the gradients at each of the given input
  ## parameters for the graph given by output.
  ##
  ## This returns a sequence of nodes, where each one calculates the gradient
  ## of the corresponding input node.
  ##
  ## Here's an example of creating an expression and it's backward graph which calculates the gradients.
  ##
  runnableExamples:
    let b = newBuilder("test")
    # forward expression
    let x = b.parameter(F32, [], "x")
    let y = b.parameter(F32, [2, 2], "y")
    let fwd = x * (x + y)
    # will return a slice with the expression to calculate grad(x) and grad(y)
    let grads = b.gradient(fwd, ["x", "y"])
    # builds a computation with 2 input parameters which will return a tuple with 3 results
    let comp = b.build b.makeTuple(fwd & grads)
    # will dump out details of each node for debugging
    echo comp

  debug &"get gradient at: {inputs}"
  let pathValue = b.one(output.dtype, output.dims)
  var dict = initTable[uint64, Node]()
  result = newSeq[Node](inputs.len)
  b.calcGrads(output, pathValue, inputs, result, dict)
  for i, grad in result:
    result[i] = grad.reshapeAs(b, inputs[i])

