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
import tensor, xla_wrapper


type
  BuilderError* = ref object of CatchableError
    ## Exception raised while building the graph. origMsg is the original status message.
    ## at indicates the node in the graph where the error occured.
    ## The repr of this node is added to the msg field. 
    origMsg*: string
    at*:      Node

  OpType* = enum
    tConst, tParam, tError                                      ## leaf nodes
    tNot, tNeg, tAbs, tExp, tFloor, tCeil, tRound, tLog,        ## 1 arg ops
    tLog1p, tLogistic, tSign, tCos, tSin, tTanh, tSqrt,         ## ..
    tRsqrt, tIsFinite, tCopy, tZerosLike, tTupleElement,        ## ..
    tReshape, tBroadcast, tBroadcastInDim, tCollapse,           ## ..
    tTranspose, tNarrow, tConvert,                              ## ..
    tReduceSum, tReduceMin, tReduceMax                          ## ..
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

  Builder* = ref BuilderObj
    ## Builder is used to construct new operations.

  Op = object
    ## The Op object owns it's xla_op so it will be freed once it goes out of scope.
    c:    xla_op

  Node* = ref object
    ## A Node is generated from each Op once it is added to the graph.
    ## The id number is the index to the nodes sequence in the Computation object and is set when the graph is built. 
    ## The shape is the output data type and dimesnions for the op. This must be fixed and known at build time.
    id*:    int
    shape*: Shape
    case kind*: OpType:
    of tParam:
      name*:     string
      paramId*:  int
    of tError:
      message*:  string
    of tTupleElement:
      index:     int
    of tReshape, tBroadcast, tCollapse, tTranspose, tNarrow, tRngUniform, tRngNormal, tReduce, tReduceSum, tReduceMin, tReduceMax:
      indices:   seq[int]
    of tBroadcastInDim:
      outSize, bcastDims: seq[int]
    else:
      discard
    args:   seq[Node]
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

proc checkBuilderError*(status: status_t, at: Node = nil) =
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

proc uid*(n: Node): uint64 = 
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

proc `[]`*(n: Node, i: Natural): Node =
  ## Get ith input to this node
  n.args[i]

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
  result.c = xla_builder_create(name)

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

proc getParamNodes(nodes: openarray[Node]): seq[Node] =
  ## Get params list of node references
  var nparams = 0
  var paramsLen = 0
  for n in nodes:
    if n.kind == tParam: 
      nparams += 1
      paramsLen = max(paramsLen, n.paramId+1)
  if nparams == 0:
    return
  if nparams != paramsLen:
    raiseError("parameter indexes should be contiguous", nodes[^1])
  result = newSeq[Node](nparams)
  for n in nodes:
    if n.kind == tParam:
      result[n.paramId] = n

proc build*(root: Node): Computation =
  ## Build a computation from the specified root operation. Should only be called once for a given graph.
  trace "new Computation"
  let b = op_builder(root.op.c)
  let status = b.build(root.op.c, result.c.addr)
  checkBuilderError(status, root)
  var node = root
  discard result.addNodes(node)
  result.params = getParamNodes(result.nodes)

proc last*(comp: Computation): Node =
  ## Last node defined in the graph
  comp.nodes[^1]

proc rawPtr*(comp: Computation): xla_computation = comp.c

proc name*(comp: Computation): string =
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

proc constant*(b: Builder, lit: Literal): Node =
  ## Create new constant from the given literal
  wrap(constant_literal(b.c, lit.rawPtr), tConst, info="literal")

proc parameter*(b: Builder, index: int, dtype: DataType, dims: openarray[int] = [], name = ""): Node =
  ## Create a new parameter with the given index and shape. If the name is blank then uses p<index> format.
  let name = if name == "": "p" & $index else: name
  withDims(dptr, dims):
    let param = parameter(b.c, index, dtype.cint, dims.len.cint, dptr, name.cstring)
    result = wrap(param, tParam, info=name)
    result.name = name
    result.paramId = index

proc makeTuple*(b: Builder, args: varargs[Node]): Node =
  ## Creates a new tuple from a list of ops.
  if args.len == 0:
    return wrap(op_tuple(b.c, nil, 0), tTuple)  
  var ops = cast[ptr xla_op](alloc(args.len*sizeOf(xla_op)))
  for i, arg in args:
    ptrOffset(ops, i)[] = arg.op.c
  result = wrap(op_tuple(b.c, ops, csize_t(args.len)), tTuple, args)
  dealloc(ops)

proc errorNode(b: xla_builder, message: string): Node =
  wrap(b.op_invalid_argument_error(message), tError, info="invalid argument")

proc errorNode(b: xla_builder, err: status_t): Node =
  let errMsg = $status_error_message(err)
  result = errorNode(b, $errMsg.cstring)
  status_free(err)

proc errorNode*(n: Node, message: string): Node =
  ## Node used to record an error e.g. due to invalid input types or shapes.
  errorNode(op_builder(n.op.c), message)


# constant builder
proc broadcast*(a: Node, dims: openarray[int]): Node
proc convert*(a: Node, dtype: DataType): Node

macro makeConstant(typ: untyped, ctyp: static string): untyped =
  let const_r0 = ident("constant_r0_" & ctyp)
  let const_r1 = ident("constant_r1_" & ctyp)
  
  result = quote do:
    proc constant*(b: Builder, value: `typ`): Node =
      ## Create a new scalar constant from the given value.
      wrap(`const_r0`(b.c, value), tConst, [], $value)

    proc constant*(b: Builder, value: openarray[`typ`]): Node =
      ## Create a new vector constant from the given value.
      let xop = `const_r1`(b.c, value[0].unsafeAddr, csize_t(value.len))
      wrap(xop, tConst, [], $value)

makeConstant(int32, "int32_t")
makeConstant(int64, "int64_t")
makeConstant(float32, "float")
makeConstant(float64, "double")

proc constant*(b: Builder, value: float, dtype: Datatype): Node =
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


template namedConstant(symbol, call, name: untyped) =
  proc `symbol`(b: xla_builder, dtype = F32, dims: openarray[int] = []): Node =
    let node = wrap(`call`(b, cint(dtype)), tConst, [], `name`)
    if dims.len > 0:
      broadcast(node, dims)
    else:
      node

  proc `symbol`*(b: Builder, dtype = F32, dims: openarray[int] = []): Node =
    ## Constant op of given type and value. If dims is given then it is brooadcast to that shape.
    `symbol`(b.c, dtype, dims)

namedConstant(zero, op_zero, "0")
namedConstant(one, op_one, "1")
namedConstant(minValue, op_min_value, "min_value")
namedConstant(maxValue, op_max_value, "max_value")


# operations
template binop(name, opname, typ: untyped): untyped =
  proc `name`*(a, b: Node): Node =
    wrap(`opname`(a.op.c, b.op.c), typ, [a, b])

binop(`+`, op_add, tAdd)
binop(`-`, op_sub, tSub)
binop(`*`, op_mul, tMul)
binop(`/`, op_div, tDiv)
binop(rem, op_rem, tRem)
binop(max, op_max, tMax)
binop(min, op_min, tMin)
binop(pow, op_pow, tPow)
binop(dot, op_dot, tDot)
binop(logicalAnd, op_and, tAnd)
binop(logicalOr, op_or, tOr)
binop(`==`, op_eq, tEq)
binop(`!=`, op_ne, tNe)
binop(`>=`, op_ge, tGe)
binop(`>`, op_gt, tGt)
binop(`<=`, op_le, tLe)
binop(`<`, op_lt, tLt)

template unary(name, opname, typ: untyped): untyped =
  proc `name`*(a: Node): Node =
    wrap(`opname`(a.op.c), typ, [a])

unary(`!`, op_not, tNot)
unary(`-`, op_neg, tNeg)
unary(abs, op_abs, tAbs)
unary(exp, op_exp, tExp)
unary(floor, op_floor, tFloor)
unary(ceil, op_ceil, tCeil)
unary(round, op_round, tRound)
unary(log, op_log, tLog)
unary(log1p, op_log1p, tLog1p)
unary(logistic, op_logistic, tLogistic)
unary(sign, op_sign, tSign)
unary(cos, op_cos, tCos)
unary(sin, op_sin, tSin)
unary(tanh, op_tanh, tTanh)
unary(sqrt, op_sqrt, tSqrt)
unary(rsqrt, op_rsqrt, tRsqrt)
unary(isFinite, op_is_finite, tIsFinite)
unary(copy, op_copy, tCopy)
unary(zerosLike, op_zeros_like, tZerosLike)

proc normalize(index, rank: int): int =
  if index < 0: rank+index else: index

proc tupleElement*(a: Node, index: int): Node =
  ## Return the element from the input tuple at index.
  result = wrap(op_get_tuple_element(a.op.c, index), tTupleElement, [a], $index)
  result.index = index

proc convert*(a: Node, dtype: DataType): Node =
  ## Convert type of elements to dtype.
  wrap(op_convert_element_type(a.op.c, cint(dtype)), tConvert, [a], $dtype)

proc reshape*(a: Node, dims: varargs[int]): Node =
  ## Reshape the input node to dims. Total number of elements is unchanged.
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  var dims2 = @dims
  let minusAt = dims.find(-1)
  if minusAt >= 0:
    var other = @dims
    other.delete(minusAt)
    let elems = prod(a.dims)
    let nother = prod(other)
    if nother > 0 and elems mod nother == 0:
      dims2[minusAt] = elems div nother
      debug &"{a.dims} reshape {dims} => {dims2}"
  withDims(dptr, dims2):
   result = wrap(op_reshape(a.op.c, csize_t(dims2.len), dptr), tReshape, [a], $dims2)
   result.indices = dims2

proc broadcast*(a: Node, dims: openarray[int]): Node =
  ## Add new leading dimensions to the input node.
  withDims(dptr, dims):
    result = wrap(op_broadcast(a.op.c, csize_t(dims.len), dptr), tBroadcast, [a], $dims)
    result.indices = @dims

proc broadcastInDim*(a: Node, outSize, bcastDims: openarray[int]): Node =
  ## Expand dims at each index in bcastDimes from 1 to the corresponding value in outSize.
  withDims(dptr1, outSize):
    withDims(dptr2, bcastDims):
      let op = op_broadcast_in_dim(a.op.c, csize_t(outSize.len), dptr1, csize_t(bcastDims.len), dptr2)
      result = wrap(op, tBroadcastInDim, [a], &"(outSize:{outSize} bcastDims:{bcastDims})")
      result.outSize = @outSize
      result.bcastDims = @bcastDims

# collapse given dimensions
proc collapse*(a: Node, dims: openarray[int]): Node =
  ## Collapse the given dimensions into a single dimension.
  ## dims should be an in-order consecutive subset of the input dims.
  withDims(dptr, dims):
    result = wrap(op_collapse(a.op.c, csize_t(dims.len), dptr), tCollapse, [a], $dims)
    result.indices = @dims

proc transpose*(a: Node, axes: varargs[int]): Node =
  ## Permute the the given axes. If no axes are given then will swap the last 2 axes.
  ## Axes indices may be negative - in this case they will be relative to the number of dimensions.
  let r = a.op.c.rank
  if r.err != nil:
    return errorNode(op_builder(a.op.c), r.err)
  var axes2: seq[int]
  if axes.len == 0 and r.val >= 2:
    axes2 = @[r.val-1, r.val-2]
  else:
    axes2 = map(axes, x => normalize(x, r.val))
  withDims(dptr, axes2):
    result = wrap(op_transpose(a.op.c, csize_t(axes2.len), dptr), tTranspose, [a], $axes2)
    result.indices = axes2

proc narrow*(a: Node, dim, start, stop: int, stride = 1): Node =
  ## Returns the data narrowed such that dimension dim ranges from start..stop-1 with step of stride.
  let op = op_slice_in_dim(a.op.c, start, stop, stride, dim)
  result = wrap(op, tNarrow, [a], &"(dim:{dim} start:{start} stop:{stop} stride:{stride})")
  result.indices = @[dim, start, stop]

proc select*(a, onTrue, onFalse: Node): Node =
  ## Select values from onTrue where a is true else from onFalse.
  wrap(op_select(a.op.c, onTrue.op.c, onFalse.op.c), tSelect, [a, onTrue, onFalse])

proc rngUniform*(minVal, maxVal: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a uniform random distribution with values from minVal to maxVal and 
  ## given dimensions. Inputs must have the same data type. This is used as the element type for the output.
  let dtype = minVal.op.c.dtype
  if dtype.err != nil:
    return errorNode(op_builder(minVal.op.c), dtype.err)
  withDims(dptr, dims):
    let op = op_rng_uniform(minVal.op.c, maxVal.op.c, cint(dtype.val), cint(dims.len), dptr)
    result = wrap(op, tRngUniform, [minVal, maxVal], $dims)
    result.indices = @dims

proc rngNormal*(mean, stddev: Node, dims: openarray[int]): Node =
  ## Generate a tensor with a normal random distribution described by mean, std deviation, 
  ## data type and dimensions. Inputs must have the same data type. This is used the as element type for the output.
  let dtype = mean.op.c.dtype
  if dtype.err != nil:
    return errorNode(op_builder(mean.op.c), dtype.err)
  withDims(dptr, dims):
    let op = op_rng_normal(mean.op.c, stddev.op.c, cint(dtype.val), cint(dims.len), dptr)
    result = wrap(op, tRngNormal, [mean, stddev], $dims)
    result.indices = @dims

proc reduce*(a, initValue: Node, comp: Computation, dims: openarray[int] = [], nodeType = tReduce): Node =
  ## Apply reduction across one or more dimensions.  i.e. comp is applied repeatedly with a pair of elements
  ## from the a input node. initValue defines the initial 'zero' value for the reduction.
  ## If no dims given then the reduction is applied across all of the input dimensions to reduce to a scalar.
  ## If the dimension index is negative then it is relative to the number of dimensions.
  let r = a.op.c.rank
  if r.err != nil:
    return errorNode(op_builder(a.op.c), r.err)
  var dims2: seq[int]
  if dims.len == 0:
    dims2 = toSeq(0 ..< r.val)
  else:
    dims2 = map(dims, x => normalize(x, r.val))
  withDims(dptr, dims2):
    let op = op_reduce(a.op.c, initValue.op.c, comp.c, dptr, csize_t(dims2.len))
    result = wrap(op, nodeType, [a, initValue], &"{comp.name}{dims2}")
    result.indices = dims2

proc reduceSum*(a: Node, dims: varargs[int]): Node =
  ## Reduce to sum of elements across one or more dimensions in the input. See reduce for details.
  let b = op_builder(a.op.c)
  let dtype = a.op.c.dtype
  if dtype.err != nil:
    return errorNode(b, dtype.err)
  let b2 = newBuilder("sum")
  let sum = build(b2.parameter(0, dtype.val) + b2.parameter(1, dtype.val))
  reduce(a, b.zero(dtype.val), sum, dims, tReduceSum)

proc reduceMin*(a: Node, dims: varargs[int]): Node =
  ## Reduce to minimum value of elements across one or more dimensions in the input. See reduce for details.
  let b = op_builder(a.op.c)
  let dtype = a.op.c.dtype
  if dtype.err != nil:
    return errorNode(b, dtype.err)
  var b2 = newBuilder("min")
  let sum = build(min(b2.parameter(0, dtype.val), b2.parameter(1, dtype.val)))
  reduce(a, b.maxValue(dtype.val), sum, dims, tReduceMin)

proc reduceMax*(a: Node, dims: varargs[int]): Node =
  ## Reduce to maximum value of elements across one or more dimensions in the input. See reduce for details.
  let b = op_builder(a.op.c)
  let dtype = a.op.c.dtype
  if dtype.err != nil:
    return errorNode(b, dtype.err)
  var b2 = newBuilder("max")
  let sum = build(max(b2.parameter(0, dtype.val), b2.parameter(1, dtype.val)))
  result = reduce(a, b.minValue(dtype.val), sum, dims, tReduceMax)


template defn(path, expression: untyped): untyped =
  GradFn(proc(path: Node): Node = expression)

proc bcast(b: Builder, v: Node, xd, yd: openarray[int]): Node =
  ## If inputs to a 2 op node are different shapes and broadcasting was applied then we need to 
  ## take account of this and cast the gradient back to the original shape when propagating it backward
  var axes: seq[int]
  let ndim = max(xd.len, yd.len)
  let xShape = repeat(1, ndim-xd.len) & @xd
  let yShape = repeat(1, ndim-yd.len) & @yd
  for i, (dx, dy) in zip(xShape, yShape):
    if dx == 1 and dy > 1: axes.add i
  if axes.len > 0:
    debug &"broadcast {v.shape.dims} => {xd}"
    v.reduce_sum(axes)
  else:
    v

proc dotgrad(n, x, y: Node): seq[GradFn] =
  ## Gradient for dot product node
  debug &"matmul: {x} x {y}"
  case n.rank
  of 0:  # vector product: [n] x [n] => []
    return @[ defn(v, v*y), defn(v, v*x) ]
  of 1:  # matrix vector product: [m, k] x [k] => [m]
    #return @[ defn(v, dot(v.appendDims(1), y.prependDims(1))), defn(v, dot(v, x)) ]
    return @[ defn(v, dot(v.reshape(-1, 1), y.reshape(1, -1))), defn(v, dot(v, x)) ]
  of 2: # matrix product: [m, k] x [k, n] => [m, n]
    return @[ defn(v, dot(v, y.transpose)), defn(v, dot(x.transpose, v)) ]
  else:
    raiseError("dot gradient not implemented for > 2 dimensions", n)

proc localGrad(b: Builder, n: Node): seq[GradFn] =
  ## Returns the gradients at each of the inputs to node as a function of the value accumulated so far.
  ## Will raise a BuilderError exception if the node type is not supported.
  case n.kind
  of tAdd:
    return @[ 
      defn(v, b.bcast(v, n[0].dims, n[1].dims)), 
      defn(v, b.bcast(v, n[1].dims, n[0].dims))
    ]
  of tSub:
    return @[ 
      defn(v, b.bcast(v, n[0].dims, n[1].dims)),
      defn(v, b.bcast(-v, n[1].dims, n[0].dims))
    ]
  of tMul:
    return @[ 
      defn(v, b.bcast(v*n[1], n[0].dims, n[1].dims)), 
      defn(v, b.bcast(v*n[0], n[1].dims, n[0].dims))
    ]
  of tDiv:
    return @[ 
      defn(v, b.bcast(v/n[1], n[0].dims, n[1].dims)), 
      defn(v, b.bcast(-v*n[0]/(n[1]*n[1]), n[1].dims, n[0].dims))
    ]
  of tDot:
    return n.dotGrad(n[0], n[1])
  of tNeg:
    return @[ defn(v, -v) ]
  of tExp:
    return @[ defn(v, v*n) ]
  of tLog:
    return @[ defn(v, v / n[0]) ]
  of tLog1p:
    return @[ defn(v, v / (b.one(n[0].dtype) + n[0]) ) ]
  of tSqrt:
    return @[ defn(v, v * b.constant(0.5, n.shape.dtype) * n[0]) ]
  of tRsqrt:
    return @[ defn(v, v * b.constant(-0.5, n.shape.dtype) * n[0]) ]
  of tTanh:
    return @[ defn(v, v * (b.one(n[0].dtype) - n * n) ) ]
  of tSin:
    return @[ defn(v, cos(v)) ]
  of tCos:
    return @[ defn(v, -sin(v)) ]
  of tAbs:
    return @[ defn(v, v * sign(n[0])) ]
  of tLogistic:
    return @[ defn(v, v * n * (b.one(n[0].dtype) - n)) ]
  of tReduceSum:
    return @[ defn(v, b.zero(n[0].dtype, n[0].shape.dims) + v) ]
  of tReshape:
    return @[ defn(v, v.reshape(n[0].shape.dims) ) ]
  of tTranspose:
    return @[ defn(v, v.transpose(n[0].indices) ) ]
  of tConvert:
    return @[ defn(v, v.convert(n[0].dtype) ) ]
  else:
    raiseError("Node type not supported for autograd", n)

proc gradient*(b: Builder, output: Node, inputs: openarray[string]): Node =
  ## Generate the graph to calculate the gradients at each of the given input
  ## parameters for the graph given by output,
  ## This returns a tuple node where each element corresponds to the
  ## graph to calc the gradient at the corresponding parameter named in the
  ## input nodes array.
  let pathValue = b.one(output.shape.dtype, output.shape.dims)
  var dict = initTable[uint64, Node]()
  var res = newSeq[Node](inputs.len+1)
  res[0] = output

  proc calcGrads(node, pathValue: Node, inputs: openarray[string]) =
    # Recursively accumulate gradients from node where pathValue is prior value to this point.
    for i, fn in b.localGrad(node):
      let input = node[i]
      var grad = fn(pathValue)
      let id = input.uid
      dict[id] = if dict.hasKey(id):
        dict[id] + grad
      else:
        grad
      if input.len > 0:
        calcGrads(input, grad, inputs)
      elif input.kind == tParam and (let n = inputs.find(input.name); n >= 0):
        res[n+1] = dict[id]

  calcGrads(output, pathValue, inputs)
  b.makeTuple(res)


