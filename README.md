# NimXLA
NimXLA contains bindings for the XLA accelerated linear algebra library and associated packages for machine learning developed at Google. See [the Tensorflow XLA docs](https://www.tensorflow.org/xla) for details on XLA. This is still an early stage experiment to see if it's feasible. Expect things to break and the API to change.

## Installing
Download the tar file with the XLA shared library and includes for your platform from the Elixir Nx repo at [elixir-nx/xla/releases](https://github.com/elixir-nx/xla/releases) and unpack it to `/usr/local`. If you install the headers in a different location you'll need to update the .compile pragma in `xla_wrapper.nim` file.

This has been tested for Linux (CPU and Cuda) and MacOS (CPU only). If you are using the Cuda GPU build then you will also need to install the corresponding Cuda version. 

Also note that the libxla_extension.so library must be in your shared library search path, e.g. by copying it to `/usr/local/lib`.

## Example 
A simple example to build and executing a graph which squares the elements from a vector and adds a constant then converts the result to a matrix with column major order:

```nim
  import nimxla/[tensor, graph]
  let c = newCPUClient()
  echo c
  # first construct and compile the graph
  let b = newBuilder("example")
  let x = b.parameter(0, F32, [50])
  let sum = x * x + b.constant(10f32)
  let comp = build sum.reshape(10, 5).transpose()
  let exec = c.compile(comp)
  # then execute it on the device with the provided input and copy back the result
  let input = toTensor[float32](1..50).toLiteral
  let res = exec.run(input).toLiteral.f32
  echo res
```
See the [examples directory](https://github.com/jnb666/nimxla/tree/main/examples) for some more examples.

## Documentation
See [the documentation index](https://jnb666.github.io/nimxla/htdocs/theindex.html).

## Module structure
- [nimxla](https://jnb666.github.io/nimxla/htdocs/nimxla.html): Contains the Client type for interfacing with the CPU or GPU device, procs to copy data between host memory and device buffers and procs for compiling and executing a Computation which has been defined using the graph module. It also exports the symbols from the tensor, literal, shape and graph submodules.

- [nimxla/tensor](https://jnb666.github.io/nimxla/htdocs/tensor.html): Defines a generic host resident n dimensional array Tensor type which can be accessed from Nim.

- [nimxla/literal](https://jnb666.github.io/nimxla/htdocs/literal.html): Defines the Literal type which is a host resident tensor or tuple of tensors in a format compatible with XLA.

- [nimxla/shape](https://jnb666.github.io/nimxla/htdocs/shape.html): Defines the Shape type which describes the memory layout, i.e. element data type and array dimensions, for all the above datatypes.

- [nimxla/graph](https://jnb666.github.io/nimxla/htdocs/graph.html): Wraps the XLA Builder and Op classes and is used to construct a tree of Nodes which can then be finalised using the build function to generate a Computation. Regular arithmetic ops and math functions are overloaded so they can be used with nodes. Extra metadata is stored so that graphs can be inspected easily. The gradient function can be used to generate a graph to perform reverse mode automatic differentiation. The autodiff implementation is inspired by the python [smallpebble](https://github.com/sradc/smallpebble) project.

- [nimxla/nn](https://jnb666.github.io/nimxla/htdocs/nn.html): Provides additional higher level functions for constructing and optimizing neural network models.

- [nimxla/data](https://jnb666.github.io/nimxla/htdocs/data.html): Provides functions to load common datasets and iterate over batches of data.

- [nimxla/train](https://jnb666.github.io/nimxla/htdocs/train.html): Contains functions for training batches of data and calculating the accuracy of the predictions.

The submodules under nimxla are exported by the main package. Other internal functions and bindings to the XLA C++ library are under the nimxla/private directory. The C wrapper code here is based on the Rust bindings from [xla-rs](https://github.com/LaurentMazare/xla-rs).

## Memory management
Each object on the C++ side is wrapped in a Nim object with a corresponding destructor to free the resource. These are marked with `=copy(...) {.error.}` so that they cannot be duplicated. Where it's useful to move these around the wrapper object is private and a ref object linked to this is exported. It's recommended to use the ORC garbage collector as this will ensure they get destroyed as soon as the ref count goes to zero.

## Dependencies
Core modules depend on XLA headers and shared library only. data module uses [zippy](https://github.com/guzba/zippy) for gzip uncompress. Examples use [cligen](https://github.com/c-blake/cligen) for command line argument parsing.

## TODO
- complete autograd for all of the defined ops
- additonal op types: convolutions, control flow etc.
- additional module types, optimizers etc.

