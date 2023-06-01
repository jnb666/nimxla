## The shape module wraps the XLA Shape type which holds the information on the data layout for any host or device buffers.

import std/[strutils, sequtils, sugar]
import private/[xla_wrapper, utils]


type
  ElemType* = float32 or float64 or int64 or int32 or uint8 or bool
  ## Allowed types for native nim Tensors

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


proc arrayShape*(dtype: DataType, dims: varargs[int]): Shape =
  ## Create a shape for an nd array.
  Shape(kind: ArrayKind, dtype: dtype, dims: @dims)

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