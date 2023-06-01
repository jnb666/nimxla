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
    tConst, tParam, tError, tIota                               ## leaf nodes
    tNot, tNeg, tAbs, tExp, tFloor, tCeil, tRound, tLog,        ## 1 arg ops
    tLog1p, tLogistic, tSign, tCos, tSin, tTanh, tSqrt,         ## ..
    tRsqrt, tIsFinite, tCopy, tZerosLike, tTupleElement,        ## ..
    tReshape, tBroadcast, tBroadcastInDim, tCollapse,           ## ..
    tTranspose, tNarrow, tConvert, tSoftmax                     ## ..
    tReduceSum, tReduceMin, tReduceMax, tArgmin, tArgmax        ## ..
    tAdd, tSub, tMul, tDiv, tRem, tMax, tMin, tPow, tDot,       ## 2 arg ops
    tAnd, tOr, tEq, tNe, tGe, tGt, tLe, tLt, tRngUniform,       ## ..
    tRngNormal, tReduce                                         ## ..
    tSelect, tTuple                                             ## 3 or more arg ops

  Result[T] = object
    ## Val holds the result if err is nil, else err has the error status
    val: T
    err: status_t

  GradFn = proc(path: Node): Node
    ## Local gradient function for autodiff

  BuilderObj = object
    ## The Builder object owns it's xla_builder so it will be freed once it goes out of scope.
    c: xla_builder

  Builder* = ref object
    ## Builder is used to construct new operations.
    params*: seq[Node]
    obj: BuilderObj

  Op = object
    ## The Op object owns it's xla_op so it will be freed once it goes out of scope.
    c:    xla_op

  Node* = ref object
    ## A Node is generated from each Op once it is added to the graph.
    ## The id number is the index to the nodes sequence in the Computation object and is set when the graph is built. 
    ## The shape is the output data type and dimesnions for the op. This must be fixed and known at build time.
    id*:    int
    shape*: Shape
    args*:  seq[Node]
    case kind*: OpType:
    of tParam:
      name*:     string
      paramId*:  int
    of tError:
      message*:  string
    of tTupleElement, tSoftmax:
      index:     int
    of tReshape, tBroadcast, tCollapse, tTranspose, tNarrow, tRngUniform, tRngNormal, tReduce, tReduceSum, 
       tReduceMin, tReduceMax, tArgmin, tArgmax:
      indices:   seq[int]
    of tBroadcastInDim:
      outSize, bcastDims: seq[int]
    else:
      discard
    info:   string
    op:     ref Op
    idnum:  uint64

  Computation* = object
    ## A Computation wraps the constructed graph after it has been finalised.
    ## The nodes sequence is a distinct list of nodes in order in which they were declared.
    ## params contatins a reference the parameters indexed by the provided index when they were defined.
    ## The Computation object owns it's xla_computation so it will be freed once it goes out of scope.
    nodes*:  seq[Node]
    params*: seq[Node]
    c: xla_computation


# memory management
proc `=copy`(a: var BuilderObj, b: BuilderObj) {.error.}
proc `=copy`(a: var Computation, b: Computation) {.error.}
proc `=copy`(dst: var Op, src: Op) {.error.}

proc `=destroy`(builder: var BuilderObj) =
  if builder.c != nil:
    trace "free Builder"
    xla_builder_free(builder.c)
    builder.c = nil

proc `=destroy`(comp: var Computation) =
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

proc res[T](val: T, err: status_t): Result[T] =
  Result[T](val: val, err: err)

proc dtype(op: xla_op): Result[DataType] =
  ## Returns the data type of the output from the op.
  let b = op_builder(op)
  var typ: cint
  let status = b.get_element_type(op, typ.addr)
  res(DataType(typ), status)

proc rank(op: xla_op): Result[int] =
  ## Returns the number of dimensions for this op or -1 if it is a tuple.
  let b = op_builder(op)
  var size: cint
  let status = b.get_dimensions_size(op, size.addr)
  res(int(size), status)

proc shape(op: xla_op): Result[Shape] =
  ## Returns the shape of the output from the op.
  let b = op_builder(op)
  var s: shape_t
  let status = b.get_shape(op, s.addr)
  if status == nil:
    res(toShape(s), nil)
  else:
    res(Shape(), status)

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

proc newBuilder*(name: string): Builder =
  ## Create a new builder which is used to generate a new graph. The name is used for debug info.
  trace "new Builder"
  new result
  result.obj.c = xla_builder_create(name)

proc wrap(xop: xla_op, typ: OpType, args: openarray[Node] = [], info = ""): Node =
  ## Wrap the raw xla_op pointer as a Node. Will throw an exception if there is an error e.g. incorrect shape.
  trace "new Op"
  result = Node(kind: typ, op: new Op, args: @args, info: info)
  result.op.c = xop
  var s = xop.shape
  if s.err == nil:
    result.shape = s.val
    s.err = op_builder(xop).get_current_status
  checkBuilderError(s.err, result)

proc addNodes(c: var Computation, node: var Node): int =
  ## Recursively add new nodes to c.nodes starting from op. Filters out repeated nodes. Returns the node id.
  for n in c.nodes:
    if n.uid == node.uid:
      assert n.id > 0
      node.id = n.id
      return n.id
  # new node we haven't seen before
  for arg in mitems(node.args):
    arg.id = c.addNodes(arg)
  node.id = c.nodes.len + 1
  c.nodes.add node
  # clear the link to the op so it can be destroyed
  node.idnum = node.uid
  node.op = nil
  return node.id

proc build*(b: Builder, root: Node): Computation =
  ## Build a computation from the specified root operation. Should only be called once for a given graph.
  trace "new Computation"
  let status = b.obj.c.build(root.op.c, result.c.addr)
  checkBuilderError(status, root)
  var node = root
  discard result.addNodes(node)
  result.params = b.params

proc last*(comp: Computation): Node =
  ## Last node defined in the graph
  comp.nodes[^1]

proc rawPtr*(comp: Computation): xla_computation = comp.c

proc name*(comp: Computation): string =
  ## Name of the computation specified when the builder was created + count of number of ops.
  $xla_computation_name(comp.c)

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
    result = wrap(param, tParam, info=name)
    result.name = name
    result.paramId = index
    b.params.add result

proc makeTuple(b: xla_builder, args: varargs[Node]): Node =
  if args.len == 0:
    return wrap(op_tuple(b, nil, 0), tTuple)  
  var ops = cast[ptr xla_op](alloc(args.len*sizeOf(xla_op)))
  for i, arg in args:
    ptrOffset(ops, i)[] = arg.op.c
  result = wrap(op_tuple(b, ops, csize_t(args.len)), tTuple, args)
  dealloc(ops)

proc makeTuple*(b: Builder, args: varargs[Node]): Node =
  ## Creates a new tuple from a list of ops.
  makeTuple(b.obj.c, args)

proc iota(b: xla_builder, dtype: DataType, dims: openarray[int], axis: int): Node =
  withDims(dptr, dims):
    let op = b.op_iota(cint(dtype), csize_t(dims.len), dptr, axis)
    result = wrap(op, tIota, info = $dims & $axis)

proc iota*(b: Builder, dtype: DataType, dims: openarray[int], axis: int): Node =
  ## Creates an array that has specified shape and holds values starting at zero and incrementing by one along 
  ## the specified axis
  b.obj.c.iota(dtype, dims, axis)

proc errorNode(b: xla_builder, message: string): Node =
  wrap(b.op_invalid_argument_error(message), tError, info="invalid argument")

proc errorNode(b: xla_builder, err: status_t): Node =
  let errMsg = $status_error_message(err)
  result = errorNode(b, $errMsg.cstring)
  status_free(err)

proc errorNode*(n: Node, message: string): Node =
  ## Node used to record an error e.g. due to invalid input types or shapes.
  errorNode(op_builder(n.op.c), message)

proc constant*(b: Builder, lit: Literal): Node =
  ## Create new constant from the given literal
  wrap(constant_literal(b.obj.c, lit.rawPtr), tConst, info="literal")

# forward defs
proc broadcast*(a: Node, dims: openarray[int]): Node
proc convert*(a: Node, dtype: DataType): Node

# constant builder
macro makeConstant(typ: untyped, ctyp: static string): untyped =
  let const_r0 = ident("constant_r0_" & ctyp)
  let const_r1 = ident("constant_r1_" & ctyp)
  let (b, value) = (ident "b", ident "value")

  result = quote do:
    proc constant*(`b`: Builder, `value`: `typ`): Node =
      ## Create a new scalar constant from the given value.
      wrap(`const_r0`(`b`.obj.c, `value`), tConst, [], $`value`)

    proc constant*(`b`: Builder, `value`: openarray[`typ`]): Node =
      ## Create a new vector constant from the given value.
      let xop = `const_r1`(`b`.obj.c, `value`[0].unsafeAddr, csize_t(`value`.len))
      wrap(xop, tConst, [], $`value`)

makeConstant(int32, "int32_t")
makeConstant(int64, "int64_t")
makeConstant(float32, "float")
makeConstant(float64, "double")

proc constant*(b: Builder, value: float, dtype: Datatype): Node =
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
    convert(b.constant(value.float32), dtype)

macro namedConstant(symbol, call, name: untyped, docs: static string) =
  let (b, dtype, dims) = (ident "b", ident "dtype", ident "dims")
  var docComment = newNimNode(nnkCommentStmt)
  docComment.strVal = docs

  result = quote do:
    proc `symbol`(`b`: xla_builder, `dtype` = F32, `dims`: openarray[int] = []): Node =
      let node = wrap(`call`(`b`, cint(`dtype`)), tConst, [], `name`)
      if `dims`.len > 0: broadcast(node, `dims`) else: node

    proc `symbol`*(`b`: Builder, `dtype` = F32, `dims`: openarray[int] = []): Node =
      `docComment`
      `symbol`(`b`.obj.c, `dtype`, `dims`)

namedConstant(zero, op_zero, "0", 
  "Create a node with the zero value for the given datatype. This is broadcast to dims if provided.")
namedConstant(one, op_one, "1", 
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
      wrap(`opname`(`a`.op.c, `b`.op.c), `typ`, [`a`, `b`])

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
      wrap(`opname`(`a`.op.c), `typ`, [`a`])

unary(`!`, op_not, tNot, "Elementwise logical not.")
unary(`-`, op_neg, tNeg, "Elementwise arithmetic negation")
unary(abs, op_abs, tAbs, "Elementwise absolute value")
unary(exp, op_exp, tExp, "Elementwise natural exponential")
unary(floor, op_floor, tFloor, "Elementwise floor rounding")
unary(ceil, op_ceil, tCeil, "Elementwise ceil rounding")
unary(round, op_round, tRound, "Elementwise nearest rounding")
unary(log, op_log, tLog, "Elementwise natural log")
unary(log1p, op_log1p, tLog1p, "Elementwise log(1 + a)")
unary(logistic, op_logistic, tLogistic, "Elementwise 1/(1 + exp(-a))")
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
  result = wrap(op_get_tuple_element(a.op.c, index), tTupleElement, [a], $index)
  result.index = index

proc convert*(a: Node, dtype: DataType): Node =
  ## Convert type of elements to dtype.
  wrap(op_convert_element_type(a.op.c, cint(dtype)), tConvert, [a], $dtype)

proc reshape*(a: Node, dims: varargs[int]): Node =
  ## Reshape the input node to dims. Total number of elements is unchanged.
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  let dims2 = reshapeDims(prod(a.dims), dims)
  withDims(dptr, dims2):
   result = wrap(op_reshape(a.op.c, csize_t(dims2.len), dptr), tReshape, [a], $dims2)
   result.indices = dims2

proc broadcast*(a: Node, dims: openarray[int]): Node =
  ## Add new leading dimensions to the input node per 
  ## [broadcast](https://www.tensorflow.org/xla/operation_semantics#broadcast)
  withDims(dptr, dims):
    result = wrap(op_broadcast(a.op.c, csize_t(dims.len), dptr), tBroadcast, [a], $dims)
    result.indices = @dims

proc broadcastInDim*(a: Node, outSize, bcastDims: openarray[int]): Node =
  ## Expand dims at each index in bcastDimes from 1 to the corresponding value in outSize as 
  ## per [broadcastInDim](https://www.tensorflow.org/xla/operation_semantics#broadcastindim)
  withDims(dptr1, outSize):
    withDims(dptr2, bcastDims):
      let op = op_broadcast_in_dim(a.op.c, csize_t(outSize.len), dptr1, csize_t(bcastDims.len), dptr2)
      result = wrap(op, tBroadcastInDim, [a], &"(outSize:{outSize} bcastDims:{bcastDims})")
      result.outSize = @outSize
      result.bcastDims = @bcastDims

# collapse given dimensions
proc collapse*(a: Node, dims: openarray[int]): Node =
  ## Collapse the given dimensions into a single dimension as per 
  ## [collapse](https://www.tensorflow.org/xla/operation_semantics#collapse).
  ## dims should be an in-order consecutive subset of the input dims.
  withDims(dptr, dims):
    result = wrap(op_collapse(a.op.c, csize_t(dims.len), dptr), tCollapse, [a], $dims)
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
    result = wrap(op_transpose(a.op.c, csize_t(axes2.len), dptr), tTranspose, [a], $axes2)
    result.indices = axes2

proc narrow*(a: Node, dim, start, stop: int, stride = 1): Node =
  ## Returns the data narrowed such that dimension dim ranges from start..stop-1 with step of stride.
  ## As per [slice](https://www.tensorflow.org/xla/operation_semantics#slice)
  let op = op_slice_in_dim(a.op.c, start, stop, stride, dim)
  result = wrap(op, tNarrow, [a], &"(dim:{dim} start:{start} stop:{stop} stride:{stride})")
  result.indices = @[dim, start, stop]

proc select*(a, onTrue, onFalse: Node): Node =
  ## Select values from onTrue where a is true else from onFalse.
  wrap(op_select(a.op.c, onTrue.op.c, onFalse.op.c), tSelect, [a, onTrue, onFalse])

proc rngUniform*(minVal, maxVal: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a uniform random distribution with values from minVal to maxVal and 
  ## given dimensions. Inputs must have the same data type. This is used as the element type for the output.
  withDims(dptr, dims):
    let op = op_rng_uniform(minVal.op.c, maxVal.op.c, cint(minVal.dtype), cint(dims.len), dptr)
    result = wrap(op, tRngUniform, [minVal, maxVal], $dims)
    result.indices = @dims

proc rngNormal*(mean, stddev: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a normal random distribution described by mean, std deviation, 
  ## data type and dimensions. Inputs must have the same data type. This is used the as element type for the output.
  withDims(dptr, dims):
    let op = op_rng_normal(mean.op.c, stddev.op.c, cint(mean.dtype), cint(dims.len), dptr)
    result = wrap(op, tRngNormal, [mean, stddev], $dims)
    result.indices = @dims

proc reduce*(a, initValue: Node, comp: Computation, dims: openarray[int] = [], 
            nodeType = tReduce, keepDims = false): Node =
  ## Apply reduction across one or more dimensions.  i.e. comp is applied repeatedly with a pair of elements
  ## from the a input node. initValue defines the initial 'zero' value for the reduction.
  ## If no dims given then the reduction is applied across all of the input dimensions to reduce to a scalar.
  ## If the dimension index is negative then it is relative to the number of dimensions.
  ## If keepDims is set then the summed dimensions are kept with a size of 1, else they are removed
  ## and the numbe of dimensions in the result is reduced.
  var shape = a.dims
  var dims2: seq[int]
  if dims.len == 0:
    dims2 = toSeq(0 ..< shape.len)
  else:
    dims2 = map(dims, x => normalize(x, shape.len))
  withDims(dptr, dims2):
    let op = op_reduce(a.op.c, initValue.op.c, comp.c, dptr, csize_t(dims2.len))
    var info = $dims2
    if keepDims: info.add ":keepDims" 
    result = wrap(op, nodeType, [a, initValue], info)
    result.indices = dims2
    if keepDims and dims2.len > 0:
      for d in dims2: shape[d] = 1
      result = result.reshape(shape)

proc sum*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to sum of elements across one or more dimensions in the input.
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = op_builder(a.op.c)
  let b2 = newBuilder("reduce")
  let sum = b2.build(b2.parameter(a.dtype) + b2.parameter(a.dtype))
  reduce(a, b.zero(a.dtype), sum, dims, tReduceSum, keepDims)

proc min*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to minimum value of elements across one or more dimensions in the input. 
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = op_builder(a.op.c)
  let b2 = newBuilder("reduce")
  let sum = b2.build(min(b2.parameter(a.dtype), b2.parameter(a.dtype)))
  reduce(a, b.maxValue(a.dtype), sum, dims, tReduceMin, keepDims)

proc max*(a: Node, dims: openarray[int] = [], keepDims = false): Node =
  ## Reduce to maximum value of elements across one or more dimensions in the input. 
  ## See `reduce<#reduce%2CNode%2CNode%2CComputation%2CopenArray%5Bint%5D>`_ for details
  let b = op_builder(a.op.c)
  let b2 = newBuilder("reduce")
  let sum = b2.build(max(b2.parameter(a.dtype), b2.parameter(a.dtype)))
  reduce(a, b.minValue(a.dtype), sum, dims, tReduceMax, keepDims)

macro argMinMax(procName, compare, initVal, opType: untyped): untyped =
  let (a, axis, keepDims, ixType) = (ident "a", ident "axis", ident "keepDims", ident "ixType")
  quote do:
    proc `procName`*(`a`: Node, `axis`: int, `keepDims` = false, `ixType` = I32): Node =
      ## Get the indices of the minimum or maxiumum values along the given axis for argmin and argmax respectively.
      ## By default the shape of the result will be as per the input with this axis removed.
      ## If keepDims is set the axis for the reduction is kept in the output with size of 1.
      ## If a negative axis is given then this is taken relative to the number of dimensions of the input.
      let b = op_builder(`a`.op.c)
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
        let op = op_reduce2(b, `a`.op.c, initValue.op.c, indexes.op.c, initIndex.op.c, comp.c, dptr, csize_t(dims.len))
        var info = $dims
        if `keepDims`: info.add ":keepDims" 
        # calc both min/max and argmin/argmax, but just use the latter
        xla_op_free(op_get_tuple_element(op, 0))
        result = wrap(op_get_tuple_element(op, 1), `opType`, [], info)
        result.indices = dims
        if `keepDims`:
          shape[axis] = 1
          result = result.reshape(shape)

argMinMax(argMax, `>=`, minValue, tArgmax)
argMinMax(argMin, `<=`, maxValue, tArgmin)

proc softmax*(a: Node, axis = -1): Node =
  ## Softmax operation, shifted for numerical stability. Will apply to the last axis by default.
  let exp_a = exp(a - a.max([axis], keepDims=true))
  let sum_a = exp_a.sum([axis], keepDims=true)
  result = exp_a / sum_a
  result.kind = tSoftmax
  result.args = @[a] 
  result.index = axis

template defn(path, expression: untyped): untyped =
  GradFn(proc(path: Node): Node = expression)

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

proc unreduce(n, x: Node): GradFn =
  ## Inverse of reduceSum operation, broadcast back to original shape
  var shape = x.dims
  for d in n.indices: shape[d] = 1
  let dims = toSeq(0 ..< shape.len)
  return defn(v, v.reshape(shape).broadcastInDim(x.dims, dims))

proc dotgrad(n, x, y: Node): seq[GradFn] =
  ## Gradient for dot product node
  case n.rank
  of 0:  # vector product: [n] x [n] => []
    return @[ defn(v, v*y), defn(v, v*x) ]
  of 1:  # matrix vector product: [m, k] x [k] => [m]
    return @[ defn(v, dot(v.reshape(-1, 1), y.reshape(1, -1))), defn(v, dot(v, x)) ]
  of 2: # matrix product: [m, k] x [k, n] => [m, n]
    return @[ defn(v, dot(v, y.transpose)), defn(v, dot(x.transpose, v)) ]
  else:
    raiseError("dot gradient not implemented for > 2 dimensions", n)

proc localGrad(b: Builder, n: Node): seq[GradFn] =
  ## Returns the gradients at each of the inputs to node as a function of the value accumulated so far.
  ## Will raise a BuilderError exception if the node type is not supported.
  var x, y: Node
  if n.len >= 1: x = n.args[0]
  if n.len >= 2: y = n.args[1]
  case n.kind
  of tAdd:
    return @[ 
      defn(v, b.unbcast(v, x.dims, y.dims)), 
      defn(v, b.unbcast(v, y.dims, x.dims))
    ]
  of tSub:
    return @[ 
      defn(v, b.unbcast(v, x.dims, y.dims)),
      defn(v, b.unbcast(-v, y.dims, x.dims))
    ]
  of tMul:
    return @[ 
      defn(v, b.unbcast(v*y, x.dims, y.dims)), 
      defn(v, b.unbcast(v*x, y.dims, x.dims))
    ]
  of tDiv:
    return @[ 
      defn(v, b.unbcast(v/y, x.dims, y.dims)), 
      defn(v, b.unbcast(-v*x/(y*y), y.dims, x.dims))
    ]
  of tDot:
    return n.dotGrad(x, y)
  of tNeg:
    return @[ defn(v, -v) ]
  of tExp:
    return @[ defn(v, v*n) ]
  of tLog:
    return @[ defn(v, v / x) ]
  of tLog1p:
    return @[ defn(v, v / (b.one(x.dtype) + x) ) ]
  of tSqrt:
    return @[ defn(v, v * b.constant(0.5, n.dtype) * x) ]
  of tRsqrt:
    return @[ defn(v, v * b.constant(-0.5, n.dtype) * x) ]
  of tTanh:
    return @[ defn(v, v * (b.one(x.dtype) - n * n) ) ]
  of tSin:
    return @[ defn(v, cos(v)) ]
  of tCos:
    return @[ defn(v, -sin(v)) ]
  of tAbs:
    return @[ defn(v, v * sign(x)) ]
  of tLogistic:
    return @[ defn(v, v * n * (b.one(x.dtype) - n)) ]
  of tReduceSum:
    return @[ unreduce(n, x) ]
  of tReshape:
    return @[ defn(v, v.reshape(x.dims) ) ]
  of tTranspose:
    return @[ defn(v, v.transpose(x.indices) ) ]
  of tConvert:
    return @[ defn(v, v.convert(x.dtype) ) ]
  of tSoftmax:
    return @[ defn(v, n-v) ]
  else:
    raiseError("Node type not supported for autograd", n)

proc calcGrads(b: Builder, node, pathValue: Node, inputs: openarray[string], 
                grads: var openarray[Node], dict: var Table[uint64, Node]) =
  ## Recursively accumulate gradients from node where pathValue is prior value to this point
  debug "grad at ", node
  for i, fn in b.localGrad(node):
    let input = node.args[i]
    var grad = fn(pathValue)
    let id = input.uid
    dict[id] = if dict.hasKey(id):
      dict[id] + grad
    else:
      grad
    if input.len > 0:
      b.calcGrads(input, grad, inputs, grads, dict)
    elif input.kind == tParam:
      let n = inputs.find(input.name)
      if n >= 0: grads[n] = dict[id]

proc gradient*(b: Builder, output: Node, inputs: openarray[string]): seq[Node] =
  ## Generate the graph to calculate the gradients at each of the given input
  ## parameters for the graph given by output,
  ## This returns an sequence of nodes, where each one calculates the gradient 
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


  let pathValue = b.one(output.dtype, output.dims)
  var dict = initTable[uint64, Node]()
  result = newSeq[Node](inputs.len)
  b.calcGrads(output, pathValue, inputs, result, dict)



