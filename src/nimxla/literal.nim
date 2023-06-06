## The literal module wraps the XLA Literal type which is a host resident tensor or tuple of tensors.
##

import std/[strutils, streams, strformat, algorithm, math]
import tensor, shape
import private/[xla_wrapper, utils]
when defined tracemem:
  import std/logging

type
  LiteralObj = object
    c: literal_t

  Literal* = ref LiteralObj
    ## Literal is reference to a tensor or tuple of tensors resident in host memory 
    ## in the format used by the XLA library.


# memory management
proc `=copy`(a: var LiteralObj, b: LiteralObj) {.error.}

proc `=destroy`(buf: var LiteralObj) =
  if buf.c != nil:
    trace &"free Literal" 
    literal_free(buf.c)
    buf.c = nil

proc newLiteral*(dtype: DataType, dims: openarray[int]): Literal =
  ## Allocate a new literal tensor with the given shape. Initialises values to zero.
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
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  trace "new Literal"
  let dims = reshapeDims(lit.len, dims)
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
                t.rawPtr, csize_t(t.len*sizeOf(T)))
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
  literal_copy_to(lit.c, result.rawPtr, csize_t(result.len*sizeOf(T)))

proc i32*(lit: Literal): Tensor[int32] =
  ## Convert to int32 Tensor. Data type must be I32.
  toTensor[int32](lit)

proc i64*(lit: Literal): Tensor[int64] =
  ## Convert to int64 Tensor. Data type must by I64.
  toTensor[int64](lit)

proc f32*(lit: Literal): Tensor[float32] = 
  ## Convert to float32 Tensor. Data type must be F32.
  toTensor[float32](lit)

proc f64*(lit: Literal): Tensor[float64] =
  ## Convert to float64 Tensor. Data type must be F64.
  toTensor[float64](lit)

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


