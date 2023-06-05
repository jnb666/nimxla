# simple example of constructing and executing a graph
import std/os
import nimxla

# pass gpu on cmd line to use Cuda GPU client
let useGpu = paramCount() >= 1 and paramStr(1) == "gpu"
# create a new client and print device info
let c = if useGpu: newGPUClient() else: newCPUClient()
echo c
# build the computation graph
let b = newBuilder("example")
let x = b.parameter(F32, [50], "x")
let sum = x * x + b^10f32
let comp = b.build sum.reshape(10, 5).transpose
# dump op info for debugging and compile to executable
echo comp
let exec = c.compile(comp)
# copy some data from the host, run the computation and copy it back
let input = toTensor[float32](1..50)
let res = exec.run([input.toLiteral])
echo res.toLiteral
