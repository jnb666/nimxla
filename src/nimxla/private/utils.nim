## A few internal definitions which are shared across multiple modules

import std/[math, atomics]
import xla_wrapper

const tracemem* {.booldefine.} = false

type
  XLAError* = object of CatchableError


proc checkError*(status: status_t) =
  ## Check error status and raises an XLAError exception if not nil.
  if status != nil:
    let message = $status_error_message(status)
    status_free(status)
    raise newException(XLAError, message)

proc ptrOffset*[T](p: ptr T, off: int): ptr T {.inline.} =
  cast[ptr T](cast[int](p) + off*sizeOf(T))

proc newChannel*[T](maxItems = 0): ptr Channel[T] =
  result = cast[ptr Channel[T]](allocShared0(sizeof(Channel[T])))
  open(result[], maxItems)

proc freeChannel*[T](ch: ptr Channel[T]) =
  ch[].close
  deallocShared(ch)

proc newAtomic*[T](val: T): ptr Atomic[T]  =
  result = cast[ptr Atomic[T]](allocShared0(sizeof(Atomic[T])))
  result[].store(val)

template trace*(msg: string): untyped =
  when tracemem:
    echo msg

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