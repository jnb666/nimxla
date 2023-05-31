## A few internal definitions which are shared across multiple modules

import std/math

const tracemem* {.booldefine.} = false

proc ptrOffset*[T](p: ptr T, off: int): ptr T {.inline.} =
  cast[ptr T](cast[int](p) + off*sizeOf(T))

template trace*(msg: string): untyped =
  when tracemem:
    debug msg

template withDims*(dptr: untyped, dims: openarray[int], code: untyped): untyped =
  block:
    var dptr: ptr int64
    if dims.len > 0:
      dptr = cast[ptr int64](dims[0].unsafeAddr)
    code

proc reshapeDims*(elems: int, dims: openarray[int]): seq[int] =
  result = @dims
  let minusAt = dims.find(-1)
  if minusAt < 0: return
  var other = @dims
  other.delete(minusAt)
  let nother = prod(other)
  if nother > 0 and elems mod nother == 0:
    result[minusAt] = elems div nother