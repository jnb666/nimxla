## The nimxla module wraps the XLA PJRT CPU and GPU client objects which are used to execute computations on a device.
## It also handles transferring data to and from the device and utilities to manage this on the host.
## See https://www.tensorflow.org/xla for an overview of the XLA library.
##
## It depends on the graph module which contains the functions to define computations from a graph of operation nodes,
## the tensor utility module and the xla_wrapper module which has the definitaions for the C bindings 
## to the XLA C++ API.
##
## Here's a simple exaxmple to build and executing a graph which squares the elements from a vector and adds a constant. 
## then converts the result to a matrix with column major order.
##

runnableExamples:
  let c = newCPUClient()
  echo c
  let b = newBuilder("example")
  let x = b.parameter(F32, [50])
  let sum = x * x + b^10f32
  let comp = b.build sum.reshape(10, 5).transpose
  let exec = c.compile(comp)
  let input = toTensor[float32](1..50).toLiteral
  let res = exec.run([input]).f32
  echo res


import std/[os, sugar, strformat, strutils, sequtils, sugar, macros, logging, tables]
import nimxla/[graph, tensor, literal, shape]
import nimxla/private/[xla_wrapper, utils]

export graph, tensor, literal, shape
export utils.XLAError


type
  LogLevel* = enum
    Info, Warning, Error, Fatal

  BufferObj = object
    c: pjrt_buffer

  Buffer* = ref BufferObj
    ## Buffer represents the device memory allocated by the client for a given tensor or tuple of tensors.

  Params* = Table[string, Buffer]
    ## Set of named input parameters for an executable call.

  ExecutableObj = object
    c:  pjrt_loaded_executable

  Executable* = object
    ## An executable is a compiled graph of operations which can be called with a defined list of parameters.
    name*:     string
    params*:   seq[string]
    outputs*:  seq[string]
    inShapes*: seq[Shape]
    outShape*: Shape
    obj: ref ExecutableObj

  ClientObj = object
    c: pjrt_client
    devs: seq[pjrt_device]

  Client* = ref ClientObj
    ## A client connects to a device such as a CPU or Cuda GPU driver and provides methods to perform computations.


# memory management
proc free(address: pointer) {.importc, header: "<stdlib.h>".}

proc `=destroy`(client: var ClientObj) =
  if client.c != nil:
    trace "free Client"
    pjrt_client_free(client.c)
    client.c = nil

proc `=destroy`(buf: var BufferObj) =
  if buf.c != nil:
    trace "free Buffer"
    pjrt_buffer_free(buf.c)
    buf.c = nil

proc `=destroy`(exec: var ExecutableObj) =
  if exec.c != nil:
    trace "free Executable"
    pjrt_loaded_executable_free(exec.c)
    exec.c = nil

proc `=copy`(a: var BufferObj, b: BufferObj) {.error.}
proc `=copy`(a: var ExecutableObj, b: ExecutableObj) {.error.}
proc `=copy`(a: var ClientObj, b: ClientObj) {.error.}


proc setLogLevel*(level: LogLevel) =
  ## Set the log level used by the Tensorflow XLA library - defaults to Info
  os.putEnv("TF_CPP_MIN_LOG_LEVEL", $int(level))

proc deviceCount*(client: Client): int =
  ## Returns number of devices associated with this client
  client.devs.len

proc platformName*(client: Client): string =
  ## Returns name of platform (CPU or Cuda)
  $pjrt_client_platform_name(client.c)

proc platformVersion*(client: Client): string =
  ## Returns version of platform (e.g. Cuda version)
  $pjrt_client_platform_version(client.c)

proc `$`*(client: Client): string =
  ## Summary of client info
  &"{client.platformName} version:{client.platformVersion} devices:{client.deviceCount}"

proc getDevices(client: var Client) =
  let n = pjrt_client_addressable_device_count(client.c)
  client.devs = newSeq[pjrt_device](n)
  pjrt_client_addressable_devices(client.c, client.devs[0].addr)

proc newCPUClient*(logLevel = Warning): Client =
  ## Create a new client for running computations on the CPU.
  setLogLevel(logLevel)
  trace "new Client"
  new result
  let status = pjrt_cpu_client_create(result.c.addr)
  checkError(status)
  result.getDevices()
  debug result

proc newGPUClient*(memoryFraction = 1.0, preallocate = false, logLevel= Warning): Client =
  ## Create a new client for running computations on the GPU using Cuda.
  ## memoryFraction limits the maximum fraction of device memory which can be allocated.
  ## If preallocate is set then this is allocated at startup.
  setLogLevel(logLevel)
  trace "new Client"
  new result
  let status = pjrt_gpu_client_create(result.c.addr, memoryFraction, preallocate)
  checkError(status)
  result.getDevices()
  debug result

proc newTPUClient*(maxInflightComputations: int, logLevel= Warning): Client =
  ## Create a new client for running computations on Google TPU accelerator.  
  setLogLevel(logLevel)
  trace "new Client"
  new result
  let status = pjrt_tpu_client_create(result.c.addr, cint(maxInflightComputations))
  checkError(status)
  result.getDevices()
  debug result


# on device buffer
proc newBuffer(buf: pjrt_buffer): Buffer =
  trace "new Buffer"
  new result
  result.c = buf

proc newBuffer*(client: Client, lit: Literal, device = 0): Buffer =
  ## Create a new buffer on the device attached to the client and copy source data from a literal value on the host
  trace "new Buffer"
  new result
  let status = pjrt_buffer_from_host_literal(client.c, client.devs[device], lit.rawPtr, result.c.addr)
  checkError(status)

proc newBuffer*(client: Client, dtype: DataType, dims: openarray[int]): Buffer =
  ## Allocate a new buffer on the device with the given shape. Initialises values to zero.
  client.newBuffer(newLiteral(dtype, dims))

proc newBuffer*[T: ElemType](client: Client, t: Tensor[T], device = 0): Buffer =
  ## Create a new buffer on the device attached to the client and copy source data from a tensor value on the host
  trace "new Buffer"
  new result
  withDims(dptr, t.dims):
    let status = pjrt_buffer_from_host_buffer(client.c, client.devs[device], t.rawPtr, cint(dtypeOf(T)), 
                                              cint(t.dims.len), dptr, result.c.addr)
    checkError(status)

proc rawPtr(buf: Buffer): pjrt_buffer = buf.c

proc shape*(buf: Buffer): Shape =
  ## Get the data type and dimenstions for the buffer.
  let s = pjrt_buffer_on_device_shape(buf.c)
  toShape(s)

proc toLiteral*(buf: Buffer): Literal =
  ## Copy buffer from device to literal on host
  trace "new Literal"
  new result
  let status = pjrt_buffer_to_literal_sync(buf.c, result.addrOf)
  checkError(status)

proc toTensor*[T: ElemType](buf: Buffer): Tensor[T] =
  ## Copy buffer from device to tensor on host
  ## Will raise an XLAError exception if the data type of source and destination do not match.
  # since the PjRtBuffer->CopyRawToHost method returns a not implemented error, goes via a literal
  result = toTensor[T](buf.toLiteral)

proc i32*(buf: Buffer): Tensor[int32] =
  ## Convert to int32 Tensor. Data type must be I32.
  toTensor[int32](buf)

proc i64*(buf: Buffer): Tensor[int64] =
  ## Convert to int64 Tensor. Data type must by I64.
  toTensor[int64](buf)

proc f32*(buf: Buffer): Tensor[float32] =
  ## Convert to float32 Tensor. Data type must be F32.
  toTensor[float32](buf)

proc f64*(buf: Buffer): Tensor[float64] =
  ## Convert to float64 Tensor. Data type must be F64.
  toTensor[float64](buf)

proc boolean*(buf: Buffer): Tensor[bool] =
  ## Convert to float64 Tensor. Data type must be Bool.
  toTensor[bool](buf)

proc `$`*(buf: Buffer): string =
  ## Print shape info
  "buffer::" & $buf.shape

proc noutputs*(exec: Executable): int =
  ## If output is a tuple then tuple length, else 1
  if exec.outShape.kind == TupleKind: exec.outShape.elems.len else: 1

proc compile*(client: Client, comp: Computation, outputs: openarray[string] = []): Executable =
  ## Compile a computation so that it can be executed on this client.
  ## outputs may optionally be specified to name the output values - they are used by the runWith method.
  trace "new Executable"
  result.obj = new ExecutableObj
  let status = compile(client.c, comp.rawPtr, result.obj.c.addr)
  if status != nil:
    let message = $status_error_message(status)
    status_free(status)
    raiseError(message, comp.last)
  result.name = comp.name
  for param in comp.params:
    result.params.add param.name
    result.inShapes.add param.shape
  result.outShape = comp.last.shape
  let nout = result.noutputs
  if outputs.len > 0:
    if outputs.len != nout:
      raiseError("incorrect number of output parameters given", comp.last)
    result.outputs = @outputs
  else:
    result.outputs = map(toSeq(0 ..< nout), x => "result" & $x)

proc `$`*(exec: Executable): string =
  # returns info on the shapes of the input and output parameters
  var params: seq[string]
  for (name, shape) in zip(exec.params, exec.inShapes):
    params.add name & ":" & $shape
  var output = $exec.outShape
  if exec.outShape.kind == TupleKind:
    let list = collect:
      for (name, shape) in zip(exec.outputs, exec.outShape.elems):
        name & ":" & $shape
    output =  "(" & list.join(", ") & ")"
  exec.name & "(" & params.join(", ") & ") => " & output

# execute kernel
template makeExecute(T: typedesc, ctype, ccall: untyped): untyped =
  proc execute(exec: Executable, args: openarray[T], outputs: ptr ptr ptr pjrt_buffer, untupleResult: bool) =
    when T is Tensor:
      var args2 = map(args, x => x.toLiteral)
    else:
      var args2 = @args
    var argp: ptr `ctype`
    if args.len > 0:
      argp = cast[ptr `ctype`](alloc(args.len * sizeOf(ctype)))
      for i, arg in args2:
        ptrOffset(argp, i)[] = arg.rawPtr
    let status = `ccall`(exec.obj.c, argp, args.len.cint, outputs, untupleResult)
    if args.len > 0:
      dealloc(argp)
    checkError(status)

makeExecute(Buffer, pjrt_buffer, execute_b)
makeExecute(Literal, literal_t, execute)

proc tupleOutputs(outputs: ptr ptr pjrt_buffer): seq[Buffer] =
  # since we don't support multiple replica sets there should be just one result with the list of unpacked resulrs
  let p1 = outputs[]
  while (let p2 = ptrOffset(p1, result.len)[]; p2 != nil):
    result.add newBuffer(p2)
  free(p1)
  free(outputs)

proc firstOutput(outputs: ptr ptr pjrt_buffer): Buffer =
  let p1 = outputs[]
  let p2 = p1[]
  result = newBuffer(p2)
  free(p1)
  free(outputs)

proc checkArgs[T: Buffer|Literal|Tensor](exec: Executable, args: openarray[T]) =
  ## check the args match the parameters set when the executable was compiled
  if args.len != exec.params.len:
    raise newException(XLAError, &"{exec.name}: expecting {exec.params.len} arguments - got {args.len}")
  for i, (shape, arg) in zip(exec.inShapes, args).pairs:
    if arg.shape != shape:
      raise newException(XLAError, &"{exec.name}: {exec.params[i]} shape should be {shape} - got {arg.shape}")

proc runAndUnpack*[T: Buffer|Literal](exec: Executable, args: openarray[T], checkShape = true): seq[Buffer] =
  ## For use where the executable returns a tuple of results.
  ## Passes the given literal or buffer arguments to the executable, launches the kernel on the associated device
  ## and returns a list of buffers unpacked from the returned tuple.
  ##
  ## By default will check that the data type and shape of the parameters matches the inputs and raise an exception
  ## if there is a mismatch. Set checkShape to false to only have the runtime check the size of the input buffers.
  if checkShape:
    checkArgs[T](exec, args)
  var outputs: ptr ptr pjrt_buffer
  execute[T](exec, args, outputs.addr, true)
  tupleOutputs(outputs)

proc runAndUnpack*(exec: Executable, checkShape = true): seq[Buffer] =
  ## Convenience method for use where the executable returns a tuple of results but does not take any input arguments.
  ## As per runAndUnpack(args)
  var outputs: ptr ptr pjrt_buffer
  var args: seq[Buffer]
  execute(exec, args, outputs.addr, true)
  tupleOutputs(outputs)

proc run*[T: Buffer|Literal](exec: Executable, args: openarray[T], checkShape = true): Buffer =
  ## Pass the given literal or buffer arguments to the executable, launch the kernel on the associated
  ## device and return a single buffer with the results. i.e. tuple results are not unpacked.
  ##
  ## By default will check that the data type and shape of the parameters matches the inputs and raise an exception
  ## there is a mismatch. Set checkShape to false to only have the runtime check the size of the input buffers.
  if checkShape:
    checkArgs[T](exec, args)
  var outputs: ptr ptr pjrt_buffer
  execute[T](exec, args, outputs.addr, false)
  firstOutput(outputs)

proc run*(exec: Executable, checkShape = true): Buffer =
  ## Convenience method for use where the executable does not take any input arguments. As per exec.run(args).
  var outputs: ptr ptr pjrt_buffer
  var args: seq[Buffer]
  execute(exec, args, outputs.addr, false)
  firstOutput(outputs)

proc initParams*(pairs: openarray[(string, Buffer)]): Params =
  ## Create new parameter list
  pairs.toTable

proc runWith*(exec: Executable, params: var Params, checkShape = true) =
  ## Run executable with given set of named input parameters.  
  ## Updates params table with outputs as named when compile was called, or using result<n> format
  ## if no names were given.
  var args: seq[Buffer]
  for name in exec.params:
    args.add params[name]
  let res = runAndUnpack(exec, args, checkShape)
  for i, name in exec.outputs:
    params[name] = res[i]

proc toLiterals*(buffers: openarray[Buffer]): seq[Literal] =
  ## Copy list of buffers back to host
  map(buffers, x => x.toLiteral)

proc tuple2*(res: openarray[Buffer]): (Buffer, Buffer) =
  ## Convenienve method to destructure list of 2 buffers to tuple
  assert res.len == 2
  (res[0], res[1])

proc tuple3*(res: openarray[Buffer]): (Buffer, Buffer, Buffer) =
  ## Convenienve method to destructure list of 3 buffers to tuple
  assert res.len == 3
  (res[0], res[1], res[2])

proc tuple4*(res: openarray[Buffer]): (Buffer, Buffer, Buffer, Buffer) =
  ## Convenienve method to destructure list of 4 buffers to tuple
  assert res.len == 4
  (res[0], res[1], res[2], res[3])

proc tuple5*(res: openarray[Buffer]): (Buffer, Buffer, Buffer, Buffer, Buffer) =
  ## Convenienve method to destructure list of 5 buffers to tuple
  assert res.len == 5
  (res[0], res[1], res[2], res[3], res[4])
