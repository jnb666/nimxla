## The tensor module contains a simple ndimensional array type which is stored in host memory. e.g.
##
runnableExamples:
  import math
  
  var t = toTensor[float32](1..6).reshape(2, 3)
  for i in 0 ..< t.dims[0]:
    t[i, 0] = sqrt(t[i, 0])
  echo t


import std/[strutils, streams, strformat, sequtils, sugar, algorithm, math, endians, strscans, macros, logging]
import private/utils
import shape

type
  TensorDataObj[T: ElemType] = object
    arr: ptr UncheckedArray[T]

  TensorData[T: ElemType] = ref TensorDataObj[T]

  Tensor*[T: ElemType] = object
    ## A Tensor is a fixed size array where the data is laid out contiguously.
    ## Copying a tensor is a shallow copy - i.e. it is a view on the same data.
    dims*:  seq[int]
    data:   TensorData[T]
    offset: int

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

  LiteralArray = object
    typ:    string
    shape:  seq[int]
    value:  NimNode


# memory management
proc `=copy`[T](a: var TensorDataObj[T], b: TensorDataObj[T]) {.error.}

proc `=destroy`[T](data: var TensorDataObj[T]) =
  if data.arr != nil:
    trace "free Tensor"
    deallocShared(data.arr)
    data.arr = nil

proc allocData[T: ElemType](size: int, zero = false): TensorData[T] =
  trace "new Tensor"
  result = new TensorData[T]
  if size > 0:
    result.arr = cast[ptr UncheckedArray[T]](
      if zero:
        allocShared0(size * sizeOf(T))
      else:
        allocShared(size * sizeOf(T))
    )

proc rawPtr*[T: ElemType](t: Tensor[T]): ptr T =
  ## Pointer to start of data buffer.
  cast [ptr T](addr t.data.arr[t.offset])

proc len*(t: Tensor): int =
  ## Number of elements in the tensor.
  prod(t.dims)

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

proc toTensor*(values: openarray[int]): Tensor[int64] =
  ## Create a new 1D int64 tensor from an array of int values
  toTensor[int64](map(values, x => x.int64))

proc toTensor*[T: ElemType](slice: HSlice[int, int]): Tensor[T] =
  ## Create a new 1D tensor of given type with elements set to the values from iterating over the slice.
  result = newTensor[T](slice.len)
  var i = 0
  for value in slice.a .. slice.b:
    result.data.arr[i] = T(value)
    i += 1

proc parseLiteral(n: NimNode, first = false): LiteralArray =
  ## helper for @@ macro
  case n.kind
  of nnkFloatLit, nnkFloat64Lit:
    return LiteralArray(typ: "float64", value: n)
  of nnkFloat32Lit:
    return LiteralArray(typ: "float32", value: n)
  of nnkIntLit:
    if first:
      return LiteralArray(typ: "int64", value: newLit(n.intVal.int64))
    else:
      return LiteralArray(typ: "int64", value: n)
  of nnkInt64Lit:
    return LiteralArray(typ: "int64", value: n)
  of nnkInt32Lit:
    return LiteralArray(typ: "int32", value: n)
  of nnkUint8Lit:
    return LiteralArray(typ: "uint8", value: n)
  of nnkIdent:
    if n.strVal == "true" or n.strVal == "false":
      return LiteralArray(typ: "bool", value: n)
  of nnkBracket:
    n.expectMinLen 1
    result.value = newNimNode(nnkBracket)
    result.shape = @[n.len]
    var childShape: seq[int]
    for i, elem in n:
      let v = parseLiteral(elem, first and i == 0)
      if i == 0:
        result.typ = v.typ
        childShape = v.shape
      elif v.shape != childShape:
        error("shape of each dimension for @@ must be uniform", n)
      if v.value.kind == nnkBracket:
        for child in v.value: result.value.add child
      else:
        result.value.add v.value
    result.shape.add childShape
    return result
  else:
    discard
  error("unsupported type for @@ - elements must be integer, float or bool literal", n)

macro `@@`*(value: untyped): untyped =
  ## Generate a tensor literal from a scalar or array e.g.
  ## ```
  ##  @@42f32               -> 0 dimensional float32 tensor
  ##  @@[1.0, 2, 3]         -> 1 dimensional float64 tensor
  ##  @@[[1, 2], [3, 4]]    -> 2 dimensional int64 tensor
  ## ```
  let v = parseLiteral(value, first=true)
  result = newCall(newTree(nnkBracketExpr, ident("toTensor"), ident(v.typ)), v.value)
  if v.shape.len > 1:
    result = newCall("reshape", result, newTree(nnkBracket, map(v.shape, x => newLit(x))))

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
  t.data.arr[t.offset+pos]

proc `[]=`*[T: ElemType](t: var Tensor, ix: varargs[int], value: T) =
  ## Update the element at the position given by the array of indices similar to [].
  let pos = index(t.dims, ix)
  assert pos < t.len
  t.data.arr[t.offset+pos] = value

proc `==`*[T: ElemType](t1, t2: Tensor[T]): bool =
  ## Checks have same shape and values are equal
  if t1.dims != t2.dims:
    return false
  for i in 0 ..< t1.len:
    if t1.data.arr[t1.offset+i] != t2.data.arr[t2.offset+i]: return false
  return true

proc approxEqual*[T: ElemType](t1, t2: Tensor[T], eps = 1e-6): bool =
  ## Checks have same shape and abs(diff) < eps for each element
  if t1.dims != t2.dims:
    return false
  for i in 0 ..< t1.len:
    if abs(t1.data.arr[t1.offset+i] - t2.data.arr[t2.offset+i]) > eps: return false
  return true

proc at*[T: ElemType](t: Tensor[T], ix: Natural): Tensor[T] =
  ## Get tensor indexed by leading dimension -  e.g. if shape of t is [2, 3, 4] then t.at(1) => [3, 4].
  ## Note that this returns a view of the data from tensor t. Use clone on the result if you need a copy.
  if t.dims.len < 1 or ix >= t.dims[0]:
    raise newException(IndexDefect, &"index {ix} out of range for at {t.dims}")
  result.dims = t.dims[1 .. ^1]
  result.offset = t.offset + ix*result.len
  result.data = t.data

proc toSeq*[T: ElemType](t: Tensor[T]): seq[T] =
  ## Copy the data associated with the tensor to a sequence of length t.len.
  result = newSeq[T](t.len)
  copyMem(result[t.offset].addr, t.data.arr, t.len*sizeOf(T))

proc clone*[T: ElemType](t: Tensor[T]): Tensor[T] =
  ## Return a copy of the tensor
  result = newTensor[T](t.dims)
  copyMem(result.data.arr, addr t.data.arr[t.offset], t.len*sizeOf(T))

proc append*[T: ELemType](t, t2: Tensor[T]): Tensor[T] =
  ## Append the data from t2 to the end of t. This will allocate and return a new tensor.
  ## New shape is [t.d0+t.d2.d0, ...] where any dimensions following the first must be the same.
  if t.dims.len < 1 or t2.dims.len < 1 or t.dims.len != t2.dims.len or t.dims[1..^1] != t2.dims[1..^1]:
    raise newException(ValueError, &"invalid dimensions for append {t.dims} {t2.dims}")
  var dims = t.dims
  dims[0] += t2.dims[0]
  result = newTensor[T](dims)
  if t.len > 0:
    copyMem(result.data.arr, t.data.arr, t.len*sizeOf(T))
  if t2.len > 0:
    copyMem(addr result.data.arr[t.len], t2.data.arr, t2.len*sizeOf(T))

proc reshape*[T: ElemType](t: Tensor[T], dims: varargs[int]): Tensor[T] =
  ## Returns a view of the same tensor with the shape changed.
  ## Number of elements in the tensor must be unchanged or will raise an exception.
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  let dims = reshapeDims(t.len, dims)
  if prod(t.dims) != prod(dims):
    raise newException(IndexDefect, "cannot reshape tensor - size is changed")
  result = t
  result.dims = @dims

proc convert*[T: ElemType](t: Tensor[T], typ: typedesc[ElemType]): Tensor[typ] =
  ## Makes a copy of the tensor with elements converted to typ.
  result = newTensor[typ](t.dims)
  for i in 0 ..< t.len:
    result.data.arr[i] = typ(t.data.arr[i])

proc readUInt32LE(s: Stream): uint32 =
  var bytes = s.readUInt32
  littleEndian32(addr result, addr bytes)

proc readUInt16LE(s: Stream): uint16 =
  var bytes = s.readUInt16
  littleEndian16(addr result, addr bytes)

proc fromNumpy(typ: string): DataType =
  let typ = typ.strip(chars = {'"', '\''})
  if typ.len < 3 or typ[0] != '<':
    return InvalidType
  case typ[1 .. ^1]
  of "f4": F32
  of "f8": F64
  of "i4": I32
  of "i8": I64
  of "u1": U8
  of "b1": Bool
  else: InvalidType

proc readTensor*[T: ElemType](s: Stream): Tensor[T] =
  ## Read tensor attributes and data from a stream. This will parse data written by write
  ## and should also handle other data as long as it is in NPY v1 or v2 format and the data
  ## type is supported. Will raise an IOError exception if the data cannot be read.
  var magic: array[6, char]
  discard s.readData(magic.addr, 6)
  if magic != ['\x93','N','U','M','P','Y']:
    raise newException(IOError, "read: invalid data format - not Numpy NPY")
  let ver = s.readUInt8
  let subver = s.readUInt8
  if (ver != 1 and ver != 2) or subver != 0:
    raise newException(IOError, &"read: invalid data version {ver}.{subver} - should be 1.0 or 2.0")
  var headerLen = if ver == 1:
    s.readUInt16LE.int
  else:
    s.readUInt32LE.int
  var header = newString(headerLen)
  discard s.readData(header[0].addr, headerLen)
  header = header.replace(" ", "").strip()
  var desc, order, shape: string
  let ok = scanf(header, "{'descr':$+,'fortran_order':$+,'shape':($+)}", desc, order, shape)
  if not ok:
    raise newException(IOError, &"read: cannot parse header '{header}'")
  let dtype = desc.fromNumpy
  if dtype == InvalidType:
    raise newException(IOError, &"read: unsupported data type in header '{desc}'")
  let expType = dtypeOf(T)
  if dtype != expType:
    raise newException(IOError, &"read: got type from data as {dtype} - expecting {expType}")
  shape = shape.strip(chars = {','})
  let dims = try:
    map(split(shape, ','), x => x.parseInt)
  except ValueError:
    raise newException(IOError, &"read: cannot parse shape from header '{shape}'")
  result = newTensor[T](dims)
  var bytes = 0
  let length = result.len * sizeOf(T)
  while bytes < length:
    bytes += s.readData(result.rawPtr, length)

proc readTensor*[T: ElemType](filename: string): Tensor[T] =
  ## Read tensor attributes and data from a file as per using the stream read function.
  let f = openFileStream(filename, mode = fmRead)
  result = readTensor[T](f)
  f.close
  debug &"read {result.shape} from {filename}"

proc write*[T: ElemType](t: Tensor[T], s: Stream) =
  ## Write tensor attributes and data to stream in Numpy NPY v1 format
  s.write "\x93NUMPY\x01\x00"
  let endian = when system.cpuEndian == littleEndian: '<' else: '>'
  let prefix = toLowerAscii(($typeOf(T))[0])
  let descr = endian & prefix & $sizeOf(T)
  # format as python tuple
  var shape = "(" & map(t.dims, x => $x).join(",")
  shape.add if t.dims.len == 1: ",)" else: ")"
  var header = &"{{'descr':'{descr}', 'fortran_order':False, 'shape':{shape}}}"
  let length = header.len + 11
  let padding = (16 - (length and 15)) and 15
  header.add spaces(padding) & '\n'
  var padLen = header.len.uint16
  var padLenLE: uint16
  littleEndian16(padLenLE.addr, padLen.addr)
  s.write padLenLE
  s.write header
  s.writeData(t.rawPtr, t.len*sizeOf(T))

proc write*[T: ElemType](t: Tensor[T], filename: string) =
  ## Write tensor attributes and data to a file in Numpy NPY v1 format.
  debug &"writing {t.shape} to {filename}"
  let f = openFileStream(filename, mode = fmWrite)
  t.write(f)
  f.close

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
    result = formatFloat(val, printOpts.floatMode, printOpts.precision)
    if printOpts.floatMode == ffDefault: result = result.trimFloat
  else:
    result = $val

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
