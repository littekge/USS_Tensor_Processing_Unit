module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x1xf32>, %arg1: tensor<4xf32>, %arg2: tensor<1x4xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1x1xf32>) -> (tensor<1x1xf32> {jax.result_info = "result[0]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [1] x [0] : (tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %2 = stablehlo.reshape %arg1 : (tensor<4xf32>) -> tensor<1x4xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x4xf32>
    %4 = stablehlo.reshape %arg2 : (tensor<1x4xf32>) -> tensor<4x1xf32>
    %5 = stablehlo.dot_general %3, %4, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<1x1xf32>
    %6 = stablehlo.reshape %arg3 : (tensor<1xf32>) -> tensor<1x1xf32>
    %7 = stablehlo.add %6, %5 : tensor<1x1xf32>
    return %7 : tensor<1x1xf32>
  }
}

