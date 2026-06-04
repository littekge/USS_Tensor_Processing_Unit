module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<6x1x3x3xf32>, %arg1: tensor<6xf32>, %arg2: tensor<16x6x3x3xf32>, %arg3: tensor<16xf32>, %arg4: tensor<120x400xf32>, %arg5: tensor<120xf32>, %arg6: tensor<84x120xf32>, %arg7: tensor<84xf32>, %arg8: tensor<10x84xf32>, %arg9: tensor<10xf32>, %arg10: tensor<1x1x28x28xf32>) -> (tensor<1x10xf32> {jax.result_info = "result[0]"}) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.convolution(%arg10, %arg0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x28x28xf32>, tensor<6x1x3x3xf32>) -> tensor<1x6x26x26xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<6xf32>) -> tensor<1x6x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x26x26xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x6x26x26xf32>
    %4 = call @relu(%3) : (tensor<1x6x26x26xf32>) -> tensor<1x6x26x26xf32>
    %5 = "stablehlo.reduce_window"(%4, %cst) <{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg11: tensor<f32>, %arg12: tensor<f32>):
      %27 = stablehlo.maximum %arg11, %arg12 : tensor<f32>
      stablehlo.return %27 : tensor<f32>
    }) : (tensor<1x6x26x26xf32>, tensor<f32>) -> tensor<1x6x13x13xf32>
    %6 = stablehlo.convolution(%5, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x6x13x13xf32>, tensor<16x6x3x3xf32>) -> tensor<1x16x11x11xf32>
    %7 = stablehlo.reshape %arg3 : (tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x16x1x1xf32>) -> tensor<1x16x11x11xf32>
    %9 = stablehlo.add %6, %8 : tensor<1x16x11x11xf32>
    %10 = call @relu_4(%9) : (tensor<1x16x11x11xf32>) -> tensor<1x16x11x11xf32>
    %11 = "stablehlo.reduce_window"(%10, %cst) <{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg11: tensor<f32>, %arg12: tensor<f32>):
      %27 = stablehlo.maximum %arg11, %arg12 : tensor<f32>
      stablehlo.return %27 : tensor<f32>
    }) : (tensor<1x16x11x11xf32>, tensor<f32>) -> tensor<1x16x5x5xf32>
    %12 = stablehlo.reshape %11 : (tensor<1x16x5x5xf32>) -> tensor<1x400xf32>
    %13 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<120x400xf32>) -> tensor<400x120xf32>
    %14 = stablehlo.dot_general %12, %13, contracting_dims = [1] x [0] : (tensor<1x400xf32>, tensor<400x120xf32>) -> tensor<1x120xf32>
    %15 = stablehlo.reshape %arg5 : (tensor<120xf32>) -> tensor<1x120xf32>
    %16 = stablehlo.add %15, %14 : tensor<1x120xf32>
    %17 = call @relu_11(%16) : (tensor<1x120xf32>) -> tensor<1x120xf32>
    %18 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<84x120xf32>) -> tensor<120x84xf32>
    %19 = stablehlo.dot_general %17, %18, contracting_dims = [1] x [0] : (tensor<1x120xf32>, tensor<120x84xf32>) -> tensor<1x84xf32>
    %20 = stablehlo.reshape %arg7 : (tensor<84xf32>) -> tensor<1x84xf32>
    %21 = stablehlo.add %20, %19 : tensor<1x84xf32>
    %22 = call @relu_20(%21) : (tensor<1x84xf32>) -> tensor<1x84xf32>
    %23 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<10x84xf32>) -> tensor<84x10xf32>
    %24 = stablehlo.dot_general %22, %23, contracting_dims = [1] x [0] : (tensor<1x84xf32>, tensor<84x10xf32>) -> tensor<1x10xf32>
    %25 = stablehlo.reshape %arg9 : (tensor<10xf32>) -> tensor<1x10xf32>
    %26 = stablehlo.add %25, %24 : tensor<1x10xf32>
    return %26 : tensor<1x10xf32>
  }
  func.func private @relu(%arg0: tensor<1x6x26x26xf32>) -> tensor<1x6x26x26xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x6x26x26xf32>
    %0 = stablehlo.maximum %arg0, %cst : tensor<1x6x26x26xf32>
    return %0 : tensor<1x6x26x26xf32>
  }
  func.func private @relu_4(%arg0: tensor<1x16x11x11xf32>) -> tensor<1x16x11x11xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x16x11x11xf32>
    %0 = stablehlo.maximum %arg0, %cst : tensor<1x16x11x11xf32>
    return %0 : tensor<1x16x11x11xf32>
  }
  func.func private @relu_11(%arg0: tensor<1x120xf32>) -> tensor<1x120xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x120xf32>
    %0 = stablehlo.maximum %arg0, %cst : tensor<1x120xf32>
    return %0 : tensor<1x120xf32>
  }
  func.func private @relu_20(%arg0: tensor<1x84xf32>) -> tensor<1x84xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x84xf32>
    %0 = stablehlo.maximum %arg0, %cst : tensor<1x84xf32>
    return %0 : tensor<1x84xf32>
  }
}

