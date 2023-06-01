## The tensor module contains a simple ndimensional array type which is stored in host memory. e.g.
##
runnableExamples:
  import math
  
  var t = toTensor[float32](1..6).reshape(2, 3)
  for i in 0 ..< t.dims[0]:
    t[i, 0] = sqrt(t[i, 0])
  echo t


import std/[strutils, streams, strformat, sequtils, algorithm, math]
import private/utils
import shape
when defined tracemem:
  import std/logging

type
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
proc `=copy`[T](a: var TensorDataObj[T], b: TensorDataObj[T]) {.error.}

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

proc rawPtr*(t: Tensor): pointer =
  ## Pointer to start of data buffer.
  cast[pointer](t.data.arr)

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
  ## If one of the dimensions is -1 then this value is inferred from the total number of elements.
  let dims = reshapeDims(t.len, dims)
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
