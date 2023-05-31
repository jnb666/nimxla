## The tensor module contains a simple ndimensional array type which is stored in host memory. e.g.
##

runnableExamples:
  import math
  
  var t = toTensor[float32](1..6).reshape(2, 3)
  for i in 0 ..< t.dims[0]:
    t[i, 0] = sqrt(t[i, 0])
  echo t

import std/[strutils,streams,strformat,sequtils,algorithm,math,sugar]
import xla_wrapper
when tracemem:
  import std/logging

type
  XLAError* = object of CatchableError

  ElemType* = float32 or float64 or int64 or int32 or uint8 or bool

  DataType* = enum
    ## DataType lists all of the XLA data types.
    InvalidType, Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, 
    Tuple, OpaqueType, C64, BF16, Token, C128

  ShapeKind* = enum ArrayKind, TupleKind

  Shape* = object
    ## A shape is gives the data type and dimensions either for a single tensor
    ## or for a tuple of tensors which might be returned from XLA. 
    case kind*: ShapeKind:
    of ArrayKind:
      dtype*: DataType
      dims*:  seq[int]
    of TupleKind:
      elems*: seq[Shape]

  LiteralObj = object
    c: literal_t

  Literal* = ref LiteralObj
    ## Literal is reference to a tensor or tuple of tensors resident in host memory 
    ## in the format used by the XLA library.

  TensorDataObj[T: ElemType] = object
    arr: ptr UncheckedArray[T]

  TensorData[T: ElemType] = ref TensorDataObj[T]

  Tensor*[T: ElemType] = object
    ## A Tensor is a fixed size array where the data is laid out contiguously.
    ## Copying a tensor is a shallow copy - i.e. it is a view on the same data.
    dims*:    seq[int]
    data:     TensorData[T]

  FormatOpts = object
    minWidth:  int
    precision: int
    floatMode: FloatFormatMode
    threshold: int
    edgeitems: int

  Formatter = object
    s:       StringStream
    indent:  int
    newline: int


# memory management
proc `=copy`(a: var LiteralObj, b: LiteralObj) {.error.}
proc `=copy`[T](a: var TensorDataObj[T], b: TensorDataObj[T]) {.error.}

proc `=destroy`(buf: var LiteralObj) =
  if buf.c != nil:
    trace &"free Literal" 
    literal_free(buf.c)
    buf.c = nil

proc `=destroy`[T](data: var TensorDataObj[T]) =
  if data.arr != nil:
    trace "free Tensor"
    deallocShared(data.arr)
    data.arr = nil

proc allocData[T: ElemType](size: int, zero = false): TensorData[T] =
  trace "new Tensor"
  result = new TensorData[T]
  result.arr = cast[ptr UncheckedArray[T]](
    if zero:
      allocShared0(size * sizeOf(T))
    else:
      allocShared(size * sizeOf(T))
  )

proc checkError*(status: status_t) =
  ## Check error status and rakses an XLAError exception if not nil.
  if status != nil:
    let message = $status_error_message(status)
    status_free(status)
    raise newException(XLAError, message)

proc len*(t: Tensor): int =
  ## Number of elements in the tensor.
  prod(t.dims)

proc rawPtr*(t: Tensor): pointer =
  cast[pointer](t.data.arr)

proc dtypeOf*(T: typedesc[ElemType]): DataType =
  ## Map from supported tensor types to XLA element type enumeration.
  when T is uint8:
    U8 
  elif T is int32:
    I32
  elif T is int64:
    I64
  elif T is float32:
    F32
  elif T is float64:
    F64
  elif T is bool:
    Bool
  else:
    raise newException(XLAError, "invalid type")

proc arrayShape*(dtype: DataType, dims: varargs[int]): Shape =
  ## Create a shape for an nd array.
  Shape(kind: ArrayKind, dtype: dtype, dims: @dims)

proc `==`*(s1, s2: Shape): bool =
  ## Have same data type and dimensions?
  if s1.kind == ArrayKind and s2.kind == ArrayKind:
    s1.dtype == s2.dtype and s1.dims == s2.dims
  elif s1.kind == TupleKind and s2.kind == TupleKind:
    s1.elems == s2.elems
  else:
    false

proc `$`*(s: Shape): string =
  ## Pretty print shape info. Will recursively generate list of shapes for tuples.
  case s.kind
  of ArrayKind:
    let name = toLowerAscii($s.dtype)
    if s.dims.len == 0:
      "<" & name & ">"
    else:
      "<" & name & " " & map(s.dims, x => $x).join(" ") & ">"
  of TupleKind:
    "(" & map(s.elems, x => $x).join(", ") & ")"

proc shape*[T: ElemType](t: Tensor[T]): Shape =
  ## Get the data type and dimensions of the tensor in XLA format.
  Shape(kind: ArrayKind, dtype: dtypeOf(T), dims: t.dims)

# constructors
proc newTensor*[T: ElemType](dims: varargs[int]): Tensor[T] =
  ## Create a new tensor with the given type and dimnsions. Data is not initialized.
  Tensor[T](dims: @dims, data: allocData[T](prod(dims)))

proc zeros*[T: ElemType](dims: varargs[int]): Tensor[T] =
  ## Create a new tensor with the given type and dimensions and elements set to zero.
  Tensor[T](dims: @dims, data: allocData[T](prod(dims), zero=true))

proc fill*[T: ElemType](dims: openarray[int], value: T): Tensor[T] =
  ## Create a new tensor withe the given type and dimensions and set every element to value. 
  result = newTensor[T](dims)
  for i in 0 ..< result.len:
    result.data.arr[i] = value

proc toTensor*[T: ElemType](value: T): Tensor[T] =
  ## Convert a scalar value of type T to a 0 dimensional tensor,
  result = newTensor[T]()
  result.data.arr[0] = value

proc toTensor*[T: ElemType](values: openarray[T]): Tensor[T] =
  ## Create a new 1D tensor of given type and size by copying data from the provided array.
  result = newTensor[T](values.len)
  copyMem(result.data.arr[0].addr, values[0].unsafeAddr, result.len*sizeOf(T))

proc toTensor*[T: ElemType](slice: HSlice[int, int]): Tensor[T] =
  ## Create a new 1D tensor of given type with elements set to the values from iterating over the slice.
  result = newTensor[T](slice.len)
  var i = 0
  for value in slice.a .. slice.b:
    result.data.arr[i] = T(value)
    i += 1

proc normalizeIndex(ix, rank: int): int =
  ## check index in range and convert -ve value to offset from end
  if ix >= 0 and ix < rank:
    return ix
  elif ix < 0 and ix + rank >= 0:
    return ix + rank
  else:
    raise newException(IndexDefect, &"index out of range: {ix} [0..<{rank}]") 

proc index(dims, pos: openarray[int]): int =
  ## index array to offset
  assert pos.len == dims.len
  var stride = 1
  for i in countdown(dims.len-1, 0):
    let ix = normalizeIndex(pos[i], dims[i])
    result += stride * ix
    stride *= dims[i]

proc `[]`*[T: ElemType](t: Tensor[T], ix: varargs[int]): T =
  ## Get the element at the position given by the array of indces or empty array for a scalar.
  ## Negative indices may be used to offset from end of the dimension.
  ## Will raise an exception if index is out of range.
  let pos = index(t.dims, ix)
  assert pos < t.len
  t.data.arr[pos]

proc `[]=`*[T: ElemType](t: var Tensor, ix: varargs[int], value: T) =
  ## Update the element at the position given by the array of indices similar to [].
  let pos = index(t.dims, ix)
  assert pos < t.len
  t.data.arr[pos] = value

proc `==`*[T: ElemType](t1, t2: Tensor[T]): bool =
  ## Checks have same shape and values are equal
  if t1.dims != t2.dims:
    return false
  for i in 0 ..< t1.len:
    if t1.data.arr[i] != t2.data.arr[i]: return false
  return true

proc toSeq*[T: ElemType](t: Tensor[T]): seq[T] =
  ## Copy the data associated with the tensor to a sequence of length t.len.
  result = newSeq[T](t.len)
  copyMem(result[0].addr, t.data.arr, t.len*sizeOf(T))

proc clone*[T: ElemType](t: Tensor[T]): Tensor[T] =
  ## Return a copy of the tensor
  result = newTensor[T](t.dims)
  copyMem(result.data.arr, t.data.arr, t.len*sizeOf(T))

proc reshape*[T: ElemType](t: Tensor[T], dims: varargs[int]): Tensor[T] =
  ## Returns a view of the same tensor with the shape changed.
  ## Number of elements in the tensor must be unchanged or will raise an exception.
  if prod(t.dims) != prod(dims):
    raise newException(IndexDefect, "cannot reshape tensor - size is changed")
  result = t
  result.dims = @dims

# formatting utils
var printOpts = FormatOpts(minWidth:8, precision:6, threshold:1000, edgeitems:4)

proc setPrintOpts*(minWidth=8, precision=6, floatMode=ffDefault, threshold=1000, edgeitems=4) =
  ## Set the formatting options for printing the tensor data with the $ operator similar to
  ## numpy.set_printoptions. See the strutils package for details of the floatMode option.
  printOpts = FormatOpts(minWidth: minWidth, precision: precision, floatMode: floatMode, 
    threshold: threshold, edgeitems: edgeitems)

func trimFloat(x: string): string =
  let sPos = find(x, '.')
  if sPos < 0: 
    return x
  var ePos = find(x, 'e', start = sPos)
  var pos = if ePos >= 0: ePos-1 else: high(x)
  while pos > sPos+1 and x[pos] == '0': pos -= 1
  if ePos < 0:
    return x[0 .. pos]
  else:
    return x[0..pos] & x[ePos..^1]

proc toString[T](val: T): string =
  when T is float64 or T is float32:
    formatFloat(val, printOpts.floatMode, printOpts.precision).trimFloat
  else:
    $val

proc format[T: ElemType](f: var Formatter, t: Tensor[T], pos: seq[int], dummy = false) =
  let nd = len(pos)
  if nd == len(t.dims):
    # output cell
    let str = if dummy: "..." else: toString(t[pos])
    f.s.write(align(str & " ", printOpts.minWidth))
    return
  # start block
  f.s.write("[")
  if nd > 0:
    f.indent += 1
  if printOpts.threshold > 0 and t.len > printOpts.threshold and 2*printOpts.edgeitems < t.dims[nd]:
    # summarize data
    for i in 0 ..< printOpts.edgeitems:
      f.format(t, pos & i, dummy)
    f.format(t, pos & printOpts.edgeitems, dummy=true)
    for i in t.dims[nd]-printOpts.edgeitems ..< t.dims[nd]:
      f.format(t, pos & i, dummy)
  else:
    # all elems in given dimension
    for i in 0 ..< t.dims[nd]:
      f.format(t, pos & i, dummy)
  # end block
  f.s.write("]")
  f.newline += 1
  if nd > 0 and pos[nd-1] < t.dims[nd-1]-1:
    f.s.write(repeat('\n', f.newline) & repeat(' ', f.indent))
    f.newline = 0
  f.indent -= 1

proc format*[T: ElemType](t: Tensor[T]): string =
  ## Pretty print summary of tensor data.
  if t.dims.len == 0:
    toString(t[])
  else:
    var f = Formatter(s: newStringStream())
    f.format(t, @[])
    f.s.data  

proc `$`*[T: ElemType](t: Tensor[T]): string =
  ## Pretty print tensor type, shape and summary of data.
  let s = t.shape
  let txt = format(t)
  if txt.contains('\n'):
    $s & "\n" & txt
  else:
    $s & txt


proc toShape*(s: shape_t, topLevel=true): Shape =
  ## Unpack the raw shape_t returned by the xla wrapper.
  let ty = DataType(s.shape_element_type)
  if ty == Tuple:
    let shapes = collect:
      for i in 0 ..< s.shape_tuple_shapes_size:
        toShape(shape_tuple_shapes(s, cint(i)), false)
    result = Shape(kind: TupleKind, elems: @shapes)
  else:
    let dims = collect:
      for i in 0 ..< s.shape_dimensions_size:
        s.shape_dimensions(i).int
    result = arrayShape(ty, dims)
  if topLevel:
    shape_free(s)

proc newLiteral*(dtype: DataType, dims: openarray[int]): Literal =
  ## Allocate a new literal tensor with the given shape.
  trace "new Literal"
  new result
  withDims(dptr, dims):
    result.c = literal_create_from_shape(cint(dtype), dptr, csize_t(dims.len))

proc rawPtr*(lit: Literal): literal_t = lit.c

proc addrOf*(lit: Literal): ptr literal_t = addr lit.c

proc shape*(lit: Literal): Shape =
  ## Get the data type and dimensions for the literal.
  var s: shape_t
  literal_shape(lit.c, s.addr)
  toShape(s)

proc len*(lit: Literal): int =
  ## Returns count of number of elements.
  literal_element_count(lit.c).int

proc dtype*(lit: Literal): DataType =
  ## Returns the element data type.
  DataType(literal_element_type(lit.c))

proc clone*(lit: Literal): Literal =
  ## Copy data to a new literal
  trace "new Literal"
  new result
  result.c = literal_clone(lit.c)

proc reshape*(lit: Literal, dims: varargs[int]): Literal =
  ## Returns a view of the same literal with the shape changed.
  ## Total number of elements in the tensor must be unchanged or will raise an exception.
  trace "new Literal"
  result = new Literal
  withDims(dptr, dims):
    let status = literal_reshape(lit.c, dptr, csize_t(dims.len), result.c.addr)
    checkError(status)

proc convert*(lit: Literal, dtype: DataType): Literal =
  ## Convert type of each element. Allocates a new literal with the result.
  ## will raise an XLAError exception if conversion fails.
  ## Just returns the orginal tensor if the data type is already matching.
  if dtype == lit.dtype: return lit
  trace "new Literal"
  result = new Literal
  let status = literal_convert(lit.c, cint(dtype), result.c.addr)
  checkError(status)

proc toLiteral*[T: ElemType](t: Tensor[T]): Literal =
  ## Create a new literal form the provided tensor shape and data.
  ## Will raise an XLAError exception if the copy fails.
  trace &"new Literal"
  new result
  withDims(dptr, t.dims):
    result.c = literal_create_from_shape_and_data(cint(dtypeOf(T)), dptr, csize_t(t.dims.len),
                t.data.arr, csize_t(t.len*sizeOf(T)))
  if result.c == nil:
    raise newException(XLAError, "error creating literal from tensor")

proc toTensor*[T: ElemType](lit: Literal): Tensor[T] =
  ## Create a new tensor form the provided literal. 
  ## Will raise an XLAError exception if the data type of source and destination do not match.
  let s = lit.shape
  let ty = dtypeOf(T)
  if ty != s.dtype:
    raise newException(XLAError, &"invalid tensor type: expected {s.dtype} - got {ty}")
  result = newTensor[T](s.dims)
  literal_copy_to(lit.c, result.data.arr, csize_t(result.len*sizeOf(T)))

proc i32*(lit: Literal): Tensor[int32] = toTensor[int32](lit)

proc i64*(lit: Literal): Tensor[int64] = toTensor[int64](lit)

proc f32*(lit: Literal): Tensor[float32] = toTensor[float32](lit)

proc f64*(lit: Literal): Tensor[float64] = toTensor[float64](lit)

proc decomposeTuple*(lit: Literal): seq[Literal] =
  ## Decompose a literal containing a tuple into a seq of literals.
  ## If input is not a tuple then returns a single element sequence.
  let s = lit.shape
  case s.kind
  of ArrayKind:
    return @[lit]
  of TupleKind:
    if s.elems.len == 0:
      return @[]
    result = newSeq[Literal](s.elems.len)
    var outputs: literal_t
    literal_decompose_tuple(lit.c, outputs.addr, csize_t(result.len))
    for i in 0 ..< result.len:
      trace "new Literal"
      result[i] = new Literal
      result[i].c = ptrOffset(outputs.addr, i)[]

proc format[T: ElemType](s: Shape, t: Tensor[T]): string =
  let txt = format(t)
  if txt.contains('\n'):
    $s & "\n" & txt
  else:
    $s & txt

proc `$`*(lit: Literal): string =
  ## Convert to a tensor and pretty print the result.
  ## will convert the data type if needed to cast to a supported tensor type.
  case lit.dtype
  of Bool:
    $toTensor[bool](lit)
  of U8:
    $toTensor[uint8](lit)
  of I8, I16, U16:
    let t = lit.convert(I32)
    format(lit.shape, toTensor[int32](t))
  of I32:
    $toTensor[int32](lit)
  of U32, U64:
    let t = lit.convert(I64)
    format(lit.shape, toTensor[int64](t))
  of I64:
    $toTensor[int64](lit)
  of F16, BF16:
    let t = lit.convert(F32)
    format(lit.shape, toTensor[float32](t))
  of F32:
    $toTensor[float32](lit)
  of F64:
    $toTensor[float64](lit)
  else:
    $lit.shape & "[...]"

template makeLiteral(typ, ccall: untyped): untyped =
  proc lit*(value: `typ`): Literal =
    ## Create new scalar literal from the given value.
    trace "new Literal"
    new result
    result.c = ccall(value)

makeLiteral(int32, create_r0_int32_t)
makeLiteral(int64, create_r0_int64_t)
makeLiteral(float32, create_r0_float)
makeLiteral(float64, create_r0_double)




