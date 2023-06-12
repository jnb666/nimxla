## This module has the definitions for the C wrapper functions used to interface with the C++ XLA compiler.
## These are just for internal use. The nimxla, graph and literal packages export the public API.
##
## In order to compile and link first unpack the XLA shared library and includes from the Elixir Nx repo at
## https://github.com/elixir-nx/xla/releases under /usr/local
## 
## The C wrapper code here is copied from the https://github.com/LaurentMazare/xla-rs Rust bindings. 

{.compile(
    "xla_wrapper.cc", 
    "-std=c++17 -Wno-deprecated-declarations -Wno-defaulted-function-deleted -I/usr/local/xla_extension/include -DLLVM_ON_UNIX=1")
.}
{.passL: "-lxla_extension -lstdc++".}

type
  struct_pjrt_client = distinct object
  struct_pjrt_device = distinct object
  struct_pjrt_buffer = distinct object
  struct_pjrt_loaded_executable = distinct object
  struct_xla_builder = distinct object
  struct_xla_computation = distinct object
  struct_xla_op = distinct object
  struct_literal = distinct object
  struct_status = distinct object
  struct_shape = distinct object

  pjrt_client* = ptr struct_pjrt_client
  pjrt_device* = ptr struct_pjrt_device
  pjrt_buffer* = ptr struct_pjrt_buffer
  pjrt_loaded_executable* = ptr struct_pjrt_loaded_executable
  xla_builder* = ptr struct_xla_builder
  xla_computation* = ptr struct_xla_computation
  xla_op* = ptr struct_xla_op
  status_t* = ptr struct_status
  shape_t* = ptr struct_shape
  literal_t* = ptr struct_literal


proc pjrt_cpu_client_create*(a1: ptr pjrt_client): status_t {.importc: "pjrt_cpu_client_create".}
proc pjrt_gpu_client_create*(a1: ptr pjrt_client; a2: cdouble; a3: bool): status_t {.importc: "pjrt_gpu_client_create".}
proc pjrt_tpu_client_create*(a1: ptr pjrt_client; a2: cint): status_t {.importc: "pjrt_tpu_client_create".}
proc pjrt_client_free*(a1: pjrt_client) {.importc: "pjrt_client_free".}
proc pjrt_client_device_count*(a1: pjrt_client): cint {.importc: "pjrt_client_device_count".}
proc pjrt_client_addressable_device_count*(a1: pjrt_client): cint {.importc: "pjrt_client_addressable_device_count".}
proc pjrt_client_devices*(a1: pjrt_client; a2: ptr pjrt_device) {.importc: "pjrt_client_devices".}
proc pjrt_client_addressable_devices*(a1: pjrt_client; a2: ptr pjrt_device) {.importc: "pjrt_client_addressable_devices".}
proc pjrt_client_platform_name*(a1: pjrt_client): cstring {.importc: "pjrt_client_platform_name".}
proc pjrt_client_platform_version*(a1: pjrt_client): cstring {.importc: "pjrt_client_platform_version".}

proc pjrt_loaded_executable_free*(a1: pjrt_loaded_executable) {.importc: "pjrt_loaded_executable_free".}

proc pjrt_device_id*(a1: pjrt_device): cint {.importc: "pjrt_device_id".}
proc pjrt_device_process_index*(a1: pjrt_device): cint {.importc: "pjrt_device_process_index".}
proc pjrt_device_local_hardware_id*(a1: pjrt_device): cint {.importc: "pjrt_device_local_hardware_id".}
proc pjrt_device_transfer_to_infeed*(a1: pjrt_device; a2: literal_t): status_t {.importc: "pjrt_device_transfer_to_infeed".}
proc pjrt_device_transfer_from_outfeed*(a1: pjrt_device; a2: literal_t): status_t {.importc: "pjrt_device_transfer_from_outfeed".}
proc pjrt_device_kind*(a1: pjrt_device): cstring {.importc: "pjrt_device_kind".}
proc pjrt_device_debug_string*(a1: pjrt_device): cstring {.importc: "pjrt_device_debug_string".}
proc pjrt_device_to_string*(a1: pjrt_device): cstring {.importc: "pjrt_device_to_string".}

proc pjrt_buffer_from_host_literal*(a1: pjrt_client; a2: pjrt_device; a3: literal_t; 
  a4: ptr pjrt_buffer): status_t {.importc: "pjrt_buffer_from_host_literal".}
proc pjrt_buffer_from_host_buffer*(a1: pjrt_client; a2: pjrt_device; a3: pointer; a4: 
  cint; a5: cint; a6: ptr int64; a7: ptr pjrt_buffer): status_t {.importc: "pjrt_buffer_from_host_buffer".}
proc pjrt_buffer_to_literal_sync*(a1: pjrt_buffer; a2: ptr literal_t): status_t {.importc: "pjrt_buffer_to_literal_sync".}
proc pjrt_buffer_copy_raw_to_host_sync*(a1: pjrt_buffer; a2: pointer; 
  a3: csize_t; a4: csize_t): status_t {.importc: "pjrt_buffer_copy_raw_to_host_sync".}
proc pjrt_buffer_on_device_shape*(a1: pjrt_buffer): shape_t {.importc: "pjrt_buffer_on_device_shape".}
proc pjrt_buffer_copy_to_device*(a1: pjrt_buffer; a2: pjrt_device; a3: 
  ptr pjrt_buffer): status_t {.importc: "pjrt_buffer_copy_to_device".}
proc pjrt_buffer_free*(a1: pjrt_buffer) {.importc: "pjrt_buffer_free".}

proc xla_builder_create*(a1: cstring): xla_builder {.importc: "xla_builder_create".}
proc xla_builder_free*(a1: xla_builder) {.importc: "xla_builder_free".}
proc constant_literal*(a1: xla_builder; a2: literal_t): xla_op {.importc: "constant_literal".}
proc parameter*(a1: xla_builder; a2: int64; a3: cint; a4: cint; a5: ptr int64; a6: cstring): xla_op {.importc: "parameter".}
proc parameter_s*(a1: xla_builder; a2: int64; a3: shape_t; a4: cstring): xla_op {.importc: "parameter_s".}
proc infeed*(a1: xla_builder; a2: cint; a3: cint; a4: ptr int64; a5: cstring): xla_op {.importc: "infeed".}
proc outfeed*(a1: xla_op; a2: cint; a3: cint; a4: ptr int64; a5: cstring) {.importc: "outfeed".}
proc get_shape*(a1: xla_builder; a2: xla_op; a3: ptr shape_t): status_t {.importc: "get_shape".}
proc get_element_type*(a1: xla_builder; a2: xla_op; a3: ptr cint): status_t {.importc: "get_element_type".}
proc get_dimensions_size*(a1: xla_builder; a2: xla_op; a3: ptr cint): status_t {.importc: "get_dimensions_size".}
proc get_dimensions*(a1: xla_builder; a2: xla_op; a3: ptr csize_t): status_t {.importc: "get_dimensions".}
proc build*(a1: xla_builder; a2: xla_op; a3: ptr xla_computation): status_t {.importc: "build".}
proc compile*(a1: pjrt_client; a2: xla_computation; a3: ptr pjrt_loaded_executable): status_t {.importc: "compile".}
proc execute*(a1: pjrt_loaded_executable; a2: ptr literal_t; a3: cint;  
  a4: ptr ptr ptr pjrt_buffer, untuple_result: bool): status_t {.importc: "execute".}
proc execute_b*(a1: pjrt_loaded_executable; a2: ptr pjrt_buffer; a3: cint; 
  a4: ptr ptr ptr pjrt_buffer, untuple_result: bool): status_t {.importc: "execute_b".}
proc first_error*(a1: xla_builder): status_t {.importc: "first_error".}
proc get_current_status*(a1: xla_builder): status_t {.importc: "get_current_status".}
proc xla_computation_name*(a1: xla_computation): cstring {.importc: "xla_computation_name".}
proc xla_computation_free*(a1: xla_computation) {.importc: "xla_computation_free".}

proc op_add*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_add".}
proc op_sub*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_sub".}
proc op_mul*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_mul".}
proc op_div*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_div".}
proc op_rem*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_rem".}
proc op_max*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_max".}
proc op_min*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_min".}
proc op_and*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_and".}
proc op_or*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_or".}
proc op_xor*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_xor".}
proc op_atan2*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_atan2".}
proc op_pow*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_pow".}
proc op_dot*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_dot".}
proc op_dot_general*(a1: xla_op; a2: xla_op; a3: ptr int64; a4: csize_t; a5: ptr int64; a6: csize_t; 
  a7: ptr int64; a8: csize_t;a9: ptr int64; a10: csize_t): xla_op {.importc: "op_dot_general".}
proc op_conv*(a1: xla_op; a2: xla_op; a3: csize_t; a4: ptr int64; a5: ptr int64; a6: ptr int64;
  a7: ptr int64; a8: ptr int64; a9: ptr int64; a10: csize_t; a11: ptr int64; a12: ptr int64,
  a13: int64, a14: int64): xla_op {.importc: "op_conv".}
proc op_eq*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_eq".}
proc op_ne*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_ne".}
proc op_ge*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_ge".}
proc op_gt*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_gt".}
proc op_le*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_le".}
proc op_lt*(a1: xla_op; a2: xla_op): xla_op {.importc: "op_lt".}
proc op_not*(a1: xla_op): xla_op {.importc: "op_not".}
proc op_abs*(a1: xla_op): xla_op {.importc: "op_abs".}
proc op_exp*(a1: xla_op): xla_op {.importc: "op_exp".}
proc op_expm1*(a1: xla_op): xla_op {.importc: "op_expm1".}
proc op_floor*(a1: xla_op): xla_op {.importc: "op_floor".}
proc op_ceil*(a1: xla_op): xla_op {.importc: "op_ceil".}
proc op_round*(a1: xla_op): xla_op {.importc: "op_round".}
proc op_log*(a1: xla_op): xla_op {.importc: "op_log".}
proc op_log1p*(a1: xla_op): xla_op {.importc: "op_log1p".}
proc op_logistic*(a1: xla_op): xla_op {.importc: "op_logistic".}
proc op_sign*(a1: xla_op): xla_op {.importc: "op_sign".}
proc op_clz*(a1: xla_op): xla_op {.importc: "op_clz".}
proc op_cos*(a1: xla_op): xla_op {.importc: "op_cos".}
proc op_sin*(a1: xla_op): xla_op {.importc: "op_sin".}
proc op_tanh*(a1: xla_op): xla_op {.importc: "op_tanh".}
proc op_real*(a1: xla_op): xla_op {.importc: "op_real".}
proc op_imag*(a1: xla_op): xla_op {.importc: "op_imag".}
proc op_sqrt*(a1: xla_op): xla_op {.importc: "op_sqrt".}
proc op_rsqrt*(a1: xla_op): xla_op {.importc: "op_rsqrt".}
proc op_cbrt*(a1: xla_op): xla_op {.importc: "op_cbrt".}
proc op_is_finite*(a1: xla_op): xla_op {.importc: "op_is_finite".}
proc op_neg*(a1: xla_op): xla_op {.importc: "op_neg".}
proc op_lower_triangle*(a1: xla_op): xla_op {.importc: "op_lower_triangle".}
proc op_upper_triangle*(a1: xla_op): xla_op {.importc: "op_upper_triangle".}
proc op_einsum1*(a1: xla_op; a2: cstring): xla_op {.importc: "op_einsum1".}
proc op_einsum2*(a1: xla_op; a2: xla_op; a3: cstring): xla_op {.importc: "op_einsum2".}
proc op_copy*(a1: xla_op): xla_op {.importc: "op_copy".}
proc op_clone*(a1: xla_op): xla_op {.importc: "op_clone".}
proc op_zeros_like*(a1: xla_op): xla_op {.importc: "op_zeros_like".}
proc op_zero_like*(a1: xla_op): xla_op {.importc: "op_zero_like".}
proc op_zero*(a1: xla_builder; a2: cint): xla_op {.importc: "op_zero".}
proc op_one*(a1: xla_builder; a2: cint): xla_op {.importc: "op_one".}
proc op_min_value*(a1: xla_builder; a2: cint): xla_op {.importc: "op_min_value".}
proc op_max_value*(a1: xla_builder; a2: cint): xla_op {.importc: "op_max_value".}
proc op_reshape*(a1: xla_op; a2: csize_t; a3: ptr int64): xla_op {.importc: "op_reshape".}
proc op_reverse*(a1: xla_op; a2: csize_t; a3: ptr int64): xla_op {.importc: "op_reverse".}
proc op_broadcast*(a1: xla_op; a2: csize_t; a3: ptr int64): xla_op {.importc: "op_broadcast".}
proc op_broadcast_in_dim*(a1: xla_op; a2: csize_t; a3: ptr int64; 
  a4: csize_t; a5: ptr int64): xla_op {.importc: "op_broadcast_in_dim".}
proc op_collapse*(a1: xla_op; a2: csize_t; a3: ptr int64): xla_op {.importc: "op_collapse".}
proc op_transpose*(a1: xla_op; a2: csize_t; a3: ptr int64): xla_op {.importc: "op_transpose".}
proc op_clamp*(a1: xla_op; a2: xla_op; a3: xla_op): xla_op {.importc: "op_clamp".}
proc op_select*(a1: xla_op; a2: xla_op; a3: xla_op): xla_op {.importc: "op_select".}
proc op_rng_uniform*(a1: xla_op; a2: xla_op; a3: cint; a4: cint; a5: ptr int64): xla_op {.importc: "op_rng_uniform".}
proc op_rng_normal*(a1: xla_op; a2: xla_op; a3: cint; a4: cint; a5: ptr int64): xla_op {.importc: "op_rng_normal".}
proc op_slice_in_dim*(a1: xla_op; a2: int64; a3: int64; a4: int64; a5: int64): xla_op {.importc: "op_slice_in_dim".}
proc op_concat_in_dim*(a1: xla_op; a2: ptr xla_op; a3: csize_t; a4: int64): xla_op {.importc: "op_concat_in_dim".}
proc op_tuple*(a1: xla_builder; a2: ptr xla_op; a3: csize_t): xla_op {.importc: "op_tuple".}
proc op_get_tuple_element*(a1: xla_op; a2: int64): xla_op {.importc: "op_get_tuple_element".}
proc op_gather*(a1: xla_op; a2: xla_op; a3: ptr int64; a4: csize_t; a5: ptr int64; a6: csize_t; 
  a7: ptr int64; a8: csize_t; a9: int64; a10: ptr int64; a11: csize_t): xla_op {.importc: "op_gather".}
proc op_scatter*(a1: xla_op; a2: xla_op; a3: xla_op, a4: xla_computation, a5: int64, 
  a6: ptr int64; a7: csize_t; a8: ptr int64; a9: csize_t, a10: ptr int64; a11: csize_t): xla_op {.importc: "op_scatter".}
proc op_convert_element_type*(a1: xla_op; a2: cint): xla_op {.importc: "op_convert_element_type".}
proc op_dimensions_size*(a1: xla_op; a2: int64): xla_op {.importc: "op_dimensions_size".}
proc op_reduce*(a1: xla_op; a2: xla_op; a3: xla_computation; a4: ptr int64; a5: csize_t): xla_op {.importc: "op_reduce".}
proc op_reduce2*(b: xla_builder; a1: xla_op; a2: xla_op; a3: xla_op; a4: xla_op; a5: xla_computation;
  a6: ptr int64; a7: csize_t): xla_op {.importc: "op_reduce2".}
proc op_reduce_window*(a1: xla_op; a2: xla_op; a3: xla_computation; a4: csize_t; a5: ptr int64; a6: ptr int64;
  a7: csize_t, a8: ptr int64; a9: ptr int64): xla_op {.importc: "op_reduce_window".}
proc op_select_and_scatter*(a1: xla_op; a2: xla_computation; a3: csize_t; a4: ptr int64; a5: ptr int64; a6: csize_t;
  a7: ptr int64; a8: ptr int64; a9: xla_op; a10: xla_op; a11: xla_computation): xla_op {.importc: "op_select_and_scatter".}
proc op_internal_error*(a1: xla_builder; a2: cstring): xla_op {.importc: "op_internal_error".}
proc op_unknown_error*(a1: xla_builder; a2: cstring): xla_op {.importc: "op_unknown_error".}
proc op_invalid_argument_error*(a1: xla_builder; a2: cstring): xla_op {.importc: "op_invalid_argument_error".}
proc op_iota1*(a1: xla_builder; a2: cint; a3: csize_t): xla_op {.importc: "op_iota1".}
proc op_iota*(a1: xla_builder; a2: cint; a3: csize_t; a4: ptr int64; a5: int64): xla_op {.importc: "op_iota".}
proc op_while*(a1: xla_computation; a2: xla_computation; a3: xla_op): xla_op {.importc: "op_while".}
proc op_conditional*(a1: xla_op; a2: xla_op; a3: xla_computation; 
  a4: xla_op; a5: xla_computation): xla_op {.importc: "op_conditional".}
proc op_builder*(a1: xla_op): xla_builder {.importc: "op_builder".}

proc xla_op_valid*(a1: xla_op): cint {.importc: "xla_op_valid".}
proc xla_op_free*(a1: xla_op) {.importc: "xla_op_free".}

proc literal_create_from_shape*(a1: cint; a2: ptr int64; a3: csize_t): literal_t {.importc: "literal_create_from_shape".}
proc literal_create_from_shape_and_data*(a1: cint; a2: ptr int64; a3: csize_t; 
  a4: pointer; a5: csize_t): literal_t {.importc: "literal_create_from_shape_and_data".}
proc literal_clone*(a1: literal_t): literal_t {.importc: "literal_clone".}
proc literal_reshape*(a1: literal_t; a2: ptr int64; a3: csize_t; a4: ptr literal_t): status_t {.importc: "literal_reshape".}
proc literal_convert*(a1: literal_t; a2: cint; a3: ptr literal_t): status_t {.importc: "literal_convert".}
proc literal_element_count*(a1: literal_t): int64 {.importc: "literal_element_count".}
proc literal_element_type*(a1: literal_t): cint {.importc: "literal_element_type".}
proc literal_shape*(a1: literal_t; a2: ptr shape_t) {.importc: "literal_shape".}
proc literal_decompose_tuple*(a1: literal_t; a2: ptr literal_t; a3: csize_t) {.importc: "literal_decompose_tuple".}
proc literal_size_bytes*(a1: literal_t): int64 {.importc: "literal_size_bytes".}
proc literal_copy_to*(a1: literal_t; a2: pointer; a3: csize_t) {.importc: "literal_copy_to".}
proc literal_copy_from*(a1: literal_t; a2: pointer; a3: csize_t) {.importc: "literal_copy_from".}
proc literal_make_tuple*(a1: ptr literal_t; a2: csize_t): literal_t {.importc: "literal_make_tuple".}
proc literal_make_tuple_owned*(a1: ptr literal_t; a2: csize_t): literal_t {.importc: "literal_make_tuple_owned".}
proc literal_free*(a1: literal_t) {.importc: "literal_free".}

proc shape_dimensions_size*(a1: shape_t): cint {.importc: "shape_dimensions_size".}
proc shape_tuple_shapes_size*(a1: shape_t): csize_t {.importc: "shape_tuple_shapes_size".}
proc shape_tuple_shapes*(a1: shape_t; a2: cint): shape_t {.importc: "shape_tuple_shapes".}
proc shape_element_type*(a1: shape_t): cint {.importc: "shape_element_type".}
proc shape_dimensions*(a1: shape_t; a2: cint): int64 {.importc: "shape_dimensions".}
proc shape_free*(a1: shape_t) {.importc: "shape_free".}
proc make_shape_array*(a1: cint; a2: csize_t; a3: ptr int64): shape_t {.importc: "make_shape_array".}
proc make_shape_tuple*(a1: csize_t; a2: ptr shape_t): shape_t {.importc: "make_shape_tuple".}

proc status_free*(a1: status_t) {.importc: "status_free".}
proc status_error_message*(a1: status_t): cstring {.importc: "status_error_message".}

proc constant_r0_int32_t*(a0: xla_builder; a1: int32): xla_op {.cdecl, importc: "constant_r0_int32_t".}
proc constant_r1_int32_t*(a0: xla_builder; a1: ptr int32; a2: csize_t): xla_op {.cdecl, importc: "constant_r1_int32_t".}
proc create_r0_int32_t*(a0: int32): literalt {.cdecl, importc: "create_r0_int32_t".}
proc create_r1_int32_t*(a0: ptr int32; a1: csize_t): literalt {.cdecl, importc: "create_r1_int32_t".}
proc constant_r0_int64_t*(a0: xla_builder; a1: int64): xla_op {.cdecl, importc: "constant_r0_int64_t".}
proc constant_r1_int64_t*(a0: xla_builder; a1: ptr int64; a2: csize_t): xla_op {.cdecl, importc: "constant_r1_int64_t".}
proc create_r0_int64_t*(a0: int64): literalt {.cdecl, importc: "create_r0_int64_t".}
proc create_r1_int64_t*(a0: ptr int64; a1: csize_t): literalt {.cdecl, importc: "create_r1_int64_t".}
proc constant_r0_float*(a0: xla_builder; a1: cfloat): xla_op {.cdecl, importc: "constant_r0_float".}
proc constant_r1_float*(a0: xla_builder; a1: ptr cfloat; a2: csize_t): xla_op {.cdecl, importc: "constant_r1_float".}
proc create_r0_float*(a0: cfloat): literalt {.cdecl, importc: "create_r0_float".}
proc create_r1_float*(a0: ptr cfloat; a1: csize_t): literalt {.cdecl, importc: "create_r1_float".}
proc constant_r0_double*(a0: xla_builder; a1: cdouble): xla_op {.cdecl, importc: "constant_r0_double".}
proc constant_r1_double*(a0: xla_builder; a1: ptr cdouble; a2: csize_t): xla_op {.cdecl, importc: "constant_r1_double".}
proc create_r0_double*(a0: cdouble): literalt {.cdecl, importc: "create_r0_double".}
proc create_r1_double*(a0: ptr cdouble; a1: csize_t): literalt {.cdecl, importc: "create_r1_double".}

