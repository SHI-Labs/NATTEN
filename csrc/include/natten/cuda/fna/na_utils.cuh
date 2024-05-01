/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
/*! \file
    \brief Fused NA (FNA) utils.
    Holds data structures and helper functions that are templated by
    spatial rank (1d, 2d, 3d), and provide (almost) generic interfaces
    for the kernel.
*/
#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>

#include <natten/cuda/fna/gemm_kernel_utils.h>

namespace natten {
namespace cuda {
namespace fna {

struct NA1dDim {
  int32_t x;

  // Default ctor
  CUTLASS_HOST_DEVICE constexpr NA1dDim(int32_t x_) : x(x_) {}

  CUTLASS_HOST_DEVICE NA1dDim() : x(0) {}

  CUTLASS_HOST_DEVICE void operator=(const NA1dDim& other) {
    x = other.x;
  }

  // Operators
  CUTLASS_HOST_DEVICE NA1dDim operator+(NA1dDim other) const {
    return NA1dDim(x + other.x);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator*(NA1dDim other) const {
    return NA1dDim(x * other.x);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator-(NA1dDim other) const {
    return NA1dDim(x - other.x);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator/(NA1dDim other) const {
    return NA1dDim(x / other.x);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator+(int32_t other) const {
    return NA1dDim(x + other);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator*(int32_t other) const {
    return NA1dDim(x * other);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator-(int32_t other) const {
    return NA1dDim(x - other);
  }

  CUTLASS_HOST_DEVICE NA1dDim operator/(int32_t other) const {
    return NA1dDim(x / other);
  }

  // Cmp
  // CUTLASS_HOST_DEVICE bool operator<=(NA1dDim other) const {
  //   return x <= other.x;
  // }

  // CUTLASS_HOST_DEVICE bool operator<(NA1dDim other) const {
  //   return x < other.x;
  // }

  // CUTLASS_HOST_DEVICE bool operator>(NA1dDim other) const {
  //   return x > other.x;
  // }

  // CUTLASS_HOST_DEVICE bool operator>=(NA1dDim other) const {
  //   return x >= other.x;
  // }

  CUTLASS_HOST_DEVICE bool operator==(NA1dDim other) const {
    return x == other.x;
  }

  // This method is used to determine whether or not the current
  // coordinate exceeds the spatial extent. This is used in
  // the NA mask (`get_window_start`).
  // It should NOT overload the >= operator.
  CUTLASS_HOST_DEVICE NA1dDim is_overflowing(NA1dDim extent) const {
    return NA1dDim(x >= extent.x);
  }

  // CUTLASS_HOST_DEVICE bool operator<=(int32_t other) const {
  //  return x <= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator<(int32_t other) const {
  //  return x < other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>(int32_t other) const {
  //  return x > other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>=(int32_t other) const {
  //  return x >= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator==(int32_t other) const {
  //  return x == other;
  //}

  // Reduce
  CUTLASS_HOST_DEVICE int64_t sum() const {
    return (int64_t)x;
  }

  CUTLASS_HOST_DEVICE int64_t prod() const {
    return (int64_t)x;
  }

  CUTLASS_HOST_DEVICE int32_t prod32() const {
    return x;
  }
};

struct NA2dDim {
  int32_t x;
  int32_t y;

  // Default ctor
  CUTLASS_HOST_DEVICE constexpr NA2dDim(int32_t x_, int32_t y_)
      : x(x_), y(y_) {}

  CUTLASS_HOST_DEVICE NA2dDim() : x(0), y(0) {}

  CUTLASS_HOST_DEVICE void operator=(const NA2dDim& other) {
    x = other.x;
    y = other.y;
  }

  // CUTLASS_HOST_DEVICE NA2dDim(int32_t n): x(n), y(n) {}

  // Operators
  CUTLASS_HOST_DEVICE NA2dDim operator+(NA2dDim other) const {
    return NA2dDim(x + other.x, y + other.y);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator*(NA2dDim other) const {
    return NA2dDim(x * other.x, y * other.y);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator-(NA2dDim other) const {
    return NA2dDim(x - other.x, y - other.y);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator/(NA2dDim other) const {
    return NA2dDim(x / other.x, y / other.y);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator+(int32_t other) const {
    return NA2dDim(x + other, y + other);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator*(int32_t other) const {
    return NA2dDim(x * other, y * other);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator-(int32_t other) const {
    return NA2dDim(x - other, y - other);
  }

  CUTLASS_HOST_DEVICE NA2dDim operator/(int32_t other) const {
    return NA2dDim(x / other, y / other);
  }

  // Cmp
  // CUTLASS_HOST_DEVICE bool operator<=(NA2dDim other) const {
  //  return x <= other.x && y <= other.y;
  //}

  // CUTLASS_HOST_DEVICE bool operator<(NA2dDim other) const {
  //  return x < other.x && y < other.y;
  //}

  // CUTLASS_HOST_DEVICE bool operator>(NA2dDim other) const {
  //  return x > other.x && y > other.y;
  //}

  // CUTLASS_HOST_DEVICE bool operator>=(NA2dDim other) const {
  //  return x >= other.x && y >= other.y;
  //}

  CUTLASS_HOST_DEVICE bool operator==(NA2dDim other) const {
    return x == other.x && y == other.y;
  }

  // This method is used to determine whether or not the current
  // coordinate exceeds the spatial extent. This is used in
  // the NA mask (`get_window_start`).
  // It should NOT overload the >= operator.
  CUTLASS_HOST_DEVICE NA2dDim is_overflowing(NA2dDim extent) const {
    return NA2dDim(x >= extent.x, y >= extent.y);
  }

  // CUTLASS_HOST_DEVICE bool operator<=(int32_t other) const {
  //  return x <= other && y <= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator<(int32_t other) const {
  //  return x < other && y < other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>(int32_t other) const {
  //  return x > other && y > other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>=(int32_t other) const {
  //  return x >= other && y >= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator==(int32_t other) const {
  //  return x == other && y == other;
  //}

  // Reduce
  CUTLASS_HOST_DEVICE int64_t sum() const {
    return (int64_t)(x + y);
  }

  CUTLASS_HOST_DEVICE int64_t prod() const {
    return (int64_t)(x * y);
  }

  CUTLASS_HOST_DEVICE int32_t prod32() const {
    return x * y;
  }
};

struct NA3dDim {
  int32_t x;
  int32_t y;
  int32_t z;

  // Default ctor
  CUTLASS_HOST_DEVICE constexpr NA3dDim(int32_t x_, int32_t y_, int32_t z_)
      : x(x_), y(y_), z(z_) {}

  CUTLASS_HOST_DEVICE NA3dDim() : x(0), y(0), z(0) {}

  CUTLASS_HOST_DEVICE void operator=(const NA3dDim& other) {
    x = other.x;
    y = other.y;
    z = other.z;
  }

  // CUTLASS_HOST_DEVICE NA3dDim(int32_t n): x(n), y(n), z(n) {}

  // Operators
  CUTLASS_HOST_DEVICE NA3dDim operator+(NA3dDim other) const {
    return NA3dDim(x + other.x, y + other.y, z + other.z);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator*(NA3dDim other) const {
    return NA3dDim(x * other.x, y * other.y, z * other.z);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator-(NA3dDim other) const {
    return NA3dDim(x - other.x, y - other.y, z - other.z);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator/(NA3dDim other) const {
    return NA3dDim(x / other.x, y / other.y, z / other.z);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator+(int32_t other) const {
    return NA3dDim(x + other, y + other, z + other);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator*(int32_t other) const {
    return NA3dDim(x * other, y * other, z * other);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator-(int32_t other) const {
    return NA3dDim(x - other, y - other, z - other);
  }

  CUTLASS_HOST_DEVICE NA3dDim operator/(int32_t other) const {
    return NA3dDim(x / other, y / other, z / other);
  }

  // Cmp
  // CUTLASS_HOST_DEVICE bool operator<=(NA3dDim other) const {
  //  return x <= other.x && y <= other.y && z <= other.z;
  //}

  // CUTLASS_HOST_DEVICE bool operator<(NA3dDim other) const {
  //  return x < other.x && y < other.y && z < other.z;
  //}

  // CUTLASS_HOST_DEVICE bool operator>(NA3dDim other) const {
  //  return x > other.x && y > other.y && z > other.z;
  //}

  // CUTLASS_HOST_DEVICE bool operator>=(NA3dDim other) const {
  //  return x >= other.x && y >= other.y && z >= other.z;
  //}

  CUTLASS_HOST_DEVICE bool operator==(NA3dDim other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  // This method is used to determine whether or not the current
  // coordinate exceeds the spatial extent. This is used in
  // the NA mask (`get_window_start`).
  // It should NOT overload the >= operator.
  CUTLASS_HOST_DEVICE NA3dDim is_overflowing(NA3dDim extent) const {
    return NA3dDim(x >= extent.x, y >= extent.y, z >= extent.z);
  }

  // CUTLASS_HOST_DEVICE bool operator<=(int32_t other) const {
  //  return x <= other && y <= other && z <= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator<(int32_t other) const {
  //  return x < other && y < other && z < other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>(int32_t other) const {
  //  return x > other && y > other && z > other;
  //}

  // CUTLASS_HOST_DEVICE bool operator>=(int32_t other) const {
  //  return x >= other && y >= other && z >= other;
  //}

  // CUTLASS_HOST_DEVICE bool operator==(int32_t other) const {
  //  return x == other && y == other && z == other;
  //}

  // Reduce
  CUTLASS_HOST_DEVICE int64_t sum() const {
    return (int64_t)(x + y + z);
  }

  CUTLASS_HOST_DEVICE int64_t prod() const {
    return (int64_t)(x * y * z);
  }

  CUTLASS_HOST_DEVICE int32_t prod32() const {
    return x * y * z;
  }
};

template <int NADim>
struct GetDim;

template <>
struct GetDim<1> {
  using type = NA1dDim;
};

template <>
struct GetDim<2> {
  using type = NA2dDim;
};

template <>
struct GetDim<3> {
  using type = NA3dDim;
};

// data structure conversions
template <typename Dim, typename TupleType>
Dim tuple_to_na_dim(TupleType v);

template <>
inline NA1dDim tuple_to_na_dim(std::tuple<int32_t> v) {
  return NA1dDim(std::get<0>(v));
}

template <>
inline NA2dDim tuple_to_na_dim(std::tuple<int32_t, int32_t> v) {
  return NA2dDim(std::get<0>(v), std::get<1>(v));
}

template <>
inline NA3dDim tuple_to_na_dim(std::tuple<int32_t, int32_t, int32_t> v) {
  return NA3dDim(std::get<0>(v), std::get<1>(v), std::get<2>(v));
}

//// Factories
// template <int Value, int NADim>
// struct MakeDim;
//
// template <int Value>
// struct MakeDim <Value, 1> {
//  static constexpr NA1dDim value = {Value};
//};
//
// template <int Value>
// struct MakeDim <Value, 2> {
//  static constexpr NA2dDim value = {Value, Value};
//};
//
// template <int Value>
// struct MakeDim <Value, 3> {
//  static constexpr NA3dDim value = {Value, Value, Value};
//};

//// Tile mappings
// template <int TileSize, int NADim>
// struct MapTileSizeToNd;
//
//// Identity mapping for all 1D problems.
// template <int TileSize>
// struct MapTileSizeToNd<TileSize, 1> {
//  static constexpr NA1dDim value = TileSize;
//};
//
//// 2D problems
// template <>
// struct MapTileSizeToNd<32, 2> {
//  static constexpr NA2dDim value = {2, 16};
//};
//
// template <>
// struct MapTileSizeToNd<64, 2> {
//  static constexpr NA2dDim value = {4, 16};
//};
//
// template <>
// struct MapTileSizeToNd<128, 2> {
//  static constexpr NA2dDim value = {8, 16};
//};
//
//// 3D problems
// template <>
// struct MapTileSizeToNd<32, 3> {
//  static constexpr NA3dDim value = {2, 2, 8};
//};
//
// template <>
// struct MapTileSizeToNd<64, 3> {
//  static constexpr NA3dDim value = {4, 4, 4};
//};
//
// template <>
// struct MapTileSizeToNd<128, 3> {
//  static constexpr NA3dDim value = {4, 4, 8};
//};

// template <int NADim>
// struct Divmod;
//
// template <>
// struct Divmod<1>  {
//   using Dim = typename GetDim<1>::type;
//
//   cutlass::FastDivmod x;
//
//   CUTLASS_HOST_DEVICE constexpr Divmod(Dim divisor): x((int)divisor.x) {}
//
//   CUTLASS_HOST_DEVICE void operator()(Dim &quotient, Dim &remainder, Dim
//   dividend) const {
//     x(quotient.x, remainder.x, dividend.x);
//   }
// };
//
// template <>
// struct Divmod<2> {
//   using Dim = typename GetDim<2>::type;
//
//   cutlass::FastDivmod x;
//   cutlass::FastDivmod y;
//
//   CUTLASS_HOST_DEVICE constexpr Divmod(Dim divisor): x((int)divisor.x),
//   y((int)divisor.y) {}
//
//   CUTLASS_HOST_DEVICE void operator()(Dim &quotient, Dim &remainder, Dim
//   dividend) const {
//     x(quotient.x, remainder.x, dividend.x);
//     y(quotient.y, remainder.y, dividend.y);
//   }
// };
//
// template <>
// struct Divmod<3> {
//   using Dim = typename GetDim<3>::type;
//
//   cutlass::FastDivmod x;
//   cutlass::FastDivmod y;
//   cutlass::FastDivmod z;
//
//   CUTLASS_HOST_DEVICE constexpr Divmod(Dim divisor): x((int)divisor.x),
//   y((int)divisor.y) {}
//
//   CUTLASS_HOST_DEVICE void operator()(Dim &quotient, Dim &remainder, Dim
//   dividend) const {
//     x(quotient.x, remainder.x, dividend.x);
//     y(quotient.y, remainder.y, dividend.y);
//   }
// };

// template <int NADim>
// struct CoordMapper;
//
// template <>
// struct CoordMapper <1> {
//  using Dim = typename GetDim<1>::type;
//
//  CUTLASS_HOST_DEVICE CoordMapper(Dim block_shape) {}
//
//  CUTLASS_HOST_DEVICE Dim operator()(int32_t block_index) {
//    return Dim(block_index);
//  }
//};
//
// template <>
// struct CoordMapper <2> {
//  using Dim = typename GetDim<2>::type;
//
//  cutlass::FastDivmod divmod_y;
//
//  CUTLASS_HOST_DEVICE CoordMapper(Dim block_shape):
//  divmod_y((int)block_shape.y) {}
//
//  CUTLASS_HOST_DEVICE Dim operator()(int32_t block_index) {
//    int index_x, index_y;
//    divmod_y(index_x, index_y, block_index);
//    return Dim(index_x, index_y);
//  }
//};
//
// template <>
// struct CoordMapper <3> {
//  using Dim = typename GetDim<3>::type;
//
//  cutlass::FastDivmod divmod_y;
//  cutlass::FastDivmod divmod_z;
//
//  CUTLASS_HOST_DEVICE CoordMapper(Dim block_shape):
//  divmod_y((int)block_shape.y), divmod_z((int)block_shape.z) {}
//
//  CUTLASS_HOST_DEVICE Dim operator()(int32_t block_index) {
//    int index_x, index_y, index_z, index_xy;
//    divmod_z(index_xy, index_z, block_index);
//    divmod_z(index_x, index_y, index_xy);
//    return Dim(index_x, index_y, index_z);
//  }
//};

template <typename Dim>
CUTLASS_HOST_DEVICE Dim
map_index_to_coord(int32_t block_index, Dim block_shape);

template <>
CUTLASS_HOST_DEVICE NA1dDim
map_index_to_coord(int32_t block_index, NA1dDim block_shape) {
  // 1d is just linear indexing.
  return NA1dDim(block_index);
};

template <>
CUTLASS_HOST_DEVICE NA2dDim
map_index_to_coord(int32_t block_index, NA2dDim block_shape) {
  auto index_x = block_index / block_shape.y;
  return NA2dDim(index_x, block_index - index_x * block_shape.y);
}

template <>
CUTLASS_HOST_DEVICE NA3dDim
map_index_to_coord(int32_t block_index, NA3dDim block_shape) {
  auto index_z = block_index % block_shape.z;
  auto index_xy = block_index / block_shape.z;
  auto index_y = index_xy % block_shape.y;
  auto index_x = index_xy / block_shape.y;
  return NA3dDim(index_x, index_y, index_z);
}

template <typename Dim>
CUTLASS_HOST_DEVICE int32_t map_coord_to_index(Dim coord, Dim block_shape);

template <>
CUTLASS_HOST_DEVICE int32_t
map_coord_to_index(NA1dDim coord, NA1dDim block_shape) {
  // 1d is just linear indexing.
  return coord.x;
}

template <>
CUTLASS_HOST_DEVICE int32_t
map_coord_to_index(NA2dDim coord, NA2dDim block_shape) {
  return coord.x * block_shape.y + coord.y;
}

template <>
CUTLASS_HOST_DEVICE int32_t
map_coord_to_index(NA3dDim coord, NA3dDim block_shape) {
  return coord.x * block_shape.y * block_shape.z + coord.y * block_shape.z +
      coord.z;
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim fast_min(Dim lhs, Dim rhs);

template <>
CUTLASS_HOST_DEVICE NA1dDim fast_min(NA1dDim lhs, NA1dDim rhs) {
  return NA1dDim(cutlass::fast_min(lhs.x, rhs.x));
}

template <>
CUTLASS_HOST_DEVICE NA2dDim fast_min(NA2dDim lhs, NA2dDim rhs) {
  return NA2dDim(
      cutlass::fast_min(lhs.x, rhs.x), cutlass::fast_min(lhs.y, rhs.y));
}

template <>
CUTLASS_HOST_DEVICE NA3dDim fast_min(NA3dDim lhs, NA3dDim rhs) {
  return NA3dDim(
      cutlass::fast_min(lhs.x, rhs.x),
      cutlass::fast_min(lhs.y, rhs.y),
      cutlass::fast_min(lhs.z, rhs.z));
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim fast_max(Dim lhs, Dim rhs);

template <>
CUTLASS_HOST_DEVICE NA1dDim fast_max(NA1dDim lhs, NA1dDim rhs) {
  return NA1dDim(cutlass::fast_max(lhs.x, rhs.x));
}

template <>
CUTLASS_HOST_DEVICE NA2dDim fast_max(NA2dDim lhs, NA2dDim rhs) {
  return NA2dDim(
      cutlass::fast_max(lhs.x, rhs.x), cutlass::fast_max(lhs.y, rhs.y));
}

template <>
CUTLASS_HOST_DEVICE NA3dDim fast_max(NA3dDim lhs, NA3dDim rhs) {
  return NA3dDim(
      cutlass::fast_max(lhs.x, rhs.x),
      cutlass::fast_max(lhs.y, rhs.y),
      cutlass::fast_max(lhs.z, rhs.z));
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim ceil_div_dim(Dim lhs, Dim rhs) {}

template <>
CUTLASS_HOST_DEVICE NA1dDim ceil_div_dim(NA1dDim lhs, NA1dDim rhs) {
  return NA1dDim(gemm_kernel_utils::ceil_div(lhs.x, rhs.x));
}

template <>
CUTLASS_HOST_DEVICE NA2dDim ceil_div_dim(NA2dDim lhs, NA2dDim rhs) {
  return NA2dDim(
      gemm_kernel_utils::ceil_div(lhs.x, rhs.x),
      gemm_kernel_utils::ceil_div(lhs.y, rhs.y));
}

template <>
CUTLASS_HOST_DEVICE NA3dDim ceil_div_dim(NA3dDim lhs, NA3dDim rhs) {
  return NA3dDim(
      gemm_kernel_utils::ceil_div(lhs.x, rhs.x),
      gemm_kernel_utils::ceil_div(lhs.y, rhs.y),
      gemm_kernel_utils::ceil_div(lhs.z, rhs.z));
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim div_dim(Dim lhs, Dim rhs) {}

template <>
CUTLASS_HOST_DEVICE NA1dDim div_dim(NA1dDim lhs, NA1dDim rhs) {
  return NA1dDim(lhs.x / rhs.x);
}

template <>
CUTLASS_HOST_DEVICE NA2dDim div_dim(NA2dDim lhs, NA2dDim rhs) {
  return NA2dDim(lhs.x / rhs.x, lhs.y / rhs.y);
}

template <>
CUTLASS_HOST_DEVICE NA3dDim div_dim(NA3dDim lhs, NA3dDim rhs) {
  return NA3dDim(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim compute_stride(Dim input_shape, int64_t last_stride);

template <>
CUTLASS_HOST_DEVICE NA1dDim
compute_stride(NA1dDim input_shape, int64_t last_stride) {
  return NA1dDim(last_stride);
}

template <>
CUTLASS_HOST_DEVICE NA2dDim
compute_stride(NA2dDim input_shape, int64_t last_stride) {
  return NA2dDim(input_shape.y * last_stride, last_stride);
}

template <>
CUTLASS_HOST_DEVICE NA3dDim
compute_stride(NA3dDim input_shape, int64_t last_stride) {
  return NA3dDim(
      input_shape.y * input_shape.z * last_stride,
      input_shape.z * last_stride,
      last_stride);
}

template <typename Dim>
CUTLASS_HOST_DEVICE bool is_coord_within_upper_bound(
    Dim coord,
    Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_upper_bound(
    NA1dDim coord,
    NA1dDim upper_bound) {
  return coord.x < upper_bound.x;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_upper_bound(
    NA2dDim coord,
    NA2dDim upper_bound) {
  return coord.x < upper_bound.x && coord.y < upper_bound.y;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_upper_bound(
    NA3dDim coord,
    NA3dDim upper_bound) {
  return coord.x < upper_bound.x && coord.y < upper_bound.y &&
      coord.z < upper_bound.z;
}

template <typename Dim>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds(
    Dim coord,
    Dim lower_bound,
    Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds(
    NA1dDim coord,
    NA1dDim lower_bound,
    NA1dDim upper_bound) {
  return coord.x >= lower_bound.x && coord.x < upper_bound.x;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds(
    NA2dDim coord,
    NA2dDim lower_bound,
    NA2dDim upper_bound) {
  return coord.x >= lower_bound.x && coord.x < upper_bound.x &&
      coord.y >= lower_bound.y && coord.y < upper_bound.y;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds(
    NA3dDim coord,
    NA3dDim lower_bound,
    NA3dDim upper_bound) {
  return coord.x >= lower_bound.x && coord.x < upper_bound.x &&
      coord.y >= lower_bound.y && coord.y < upper_bound.y &&
      coord.z >= lower_bound.z && coord.z < upper_bound.z;
}

template <typename Dim>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds_nn(Dim coord, Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds_nn(
    NA1dDim coord,
    NA1dDim upper_bound) {
  return coord.x >= 0 && coord.x < upper_bound.x;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds_nn(
    NA2dDim coord,
    NA2dDim upper_bound) {
  return coord.x >= 0 && coord.y >= 0 && coord.x < upper_bound.x &&
      coord.y < upper_bound.y;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_within_bounds_nn(
    NA3dDim coord,
    NA3dDim upper_bound) {
  return coord.x >= 0 && coord.y >= 0 && coord.z >= 0 &&
      coord.x < upper_bound.x && coord.y < upper_bound.y &&
      coord.z < upper_bound.z;
}

template <typename Dim>
CUTLASS_HOST_DEVICE Dim align_dim_down(Dim tile, Dim upper_bound);
template <typename Dim>
CUTLASS_HOST_DEVICE Dim align_dim_up(Dim tile, Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE NA1dDim align_dim_down(NA1dDim tile, NA1dDim upper_bound) {
  return NA1dDim((tile.x / upper_bound.x) * upper_bound.x);
}
template <>
CUTLASS_HOST_DEVICE NA1dDim align_dim_up(NA1dDim tile, NA1dDim upper_bound) {
  return NA1dDim(
      ((tile.x + upper_bound.x - 1) / upper_bound.x) * upper_bound.x);
}

template <>
CUTLASS_HOST_DEVICE NA2dDim align_dim_down(NA2dDim tile, NA2dDim upper_bound) {
  return NA2dDim(
      (tile.x / upper_bound.x) * upper_bound.x,
      (tile.y / upper_bound.y) * upper_bound.y);
}
template <>
CUTLASS_HOST_DEVICE NA2dDim align_dim_up(NA2dDim tile, NA2dDim upper_bound) {
  return NA2dDim(
      ((tile.x + upper_bound.x - 1) / upper_bound.x) * upper_bound.x,
      ((tile.y + upper_bound.y - 1) / upper_bound.y) * upper_bound.y);
}

template <>
CUTLASS_HOST_DEVICE NA3dDim align_dim_down(NA3dDim tile, NA3dDim upper_bound) {
  return NA3dDim(
      (tile.x / upper_bound.x) * upper_bound.x,
      (tile.y / upper_bound.y) * upper_bound.y,
      (tile.z / upper_bound.z) * upper_bound.z);
}
template <>
CUTLASS_HOST_DEVICE NA3dDim align_dim_up(NA3dDim tile, NA3dDim upper_bound) {
  return NA3dDim(
      ((tile.x + upper_bound.x - 1) / upper_bound.x) * upper_bound.x,
      ((tile.y + upper_bound.y - 1) / upper_bound.y) * upper_bound.y,
      ((tile.z + upper_bound.z - 1) / upper_bound.z) * upper_bound.z);
}

template <typename Dim>
CUTLASS_HOST_DEVICE bool is_coord_less_than_or_equal_to(
    Dim coord,
    Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE bool is_coord_less_than_or_equal_to(
    NA1dDim coord,
    NA1dDim upper_bound) {
  return coord.x <= upper_bound.x;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_less_than_or_equal_to(
    NA2dDim coord,
    NA2dDim upper_bound) {
  return coord.x <= upper_bound.x && coord.y <= upper_bound.y;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_less_than_or_equal_to(
    NA3dDim coord,
    NA3dDim upper_bound) {
  return coord.x <= upper_bound.x && coord.y <= upper_bound.y &&
      coord.z <= upper_bound.z;
}

template <typename Dim>
CUTLASS_HOST_DEVICE bool is_coord_greater_than_or_equal_to(
    Dim coord,
    Dim upper_bound);

template <>
CUTLASS_HOST_DEVICE bool is_coord_greater_than_or_equal_to(
    NA1dDim coord,
    NA1dDim upper_bound) {
  return coord.x >= upper_bound.x;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_greater_than_or_equal_to(
    NA2dDim coord,
    NA2dDim upper_bound) {
  return coord.x >= upper_bound.x && coord.y >= upper_bound.y;
}

template <>
CUTLASS_HOST_DEVICE bool is_coord_greater_than_or_equal_to(
    NA3dDim coord,
    NA3dDim upper_bound) {
  return coord.x >= upper_bound.x && coord.y >= upper_bound.y &&
      coord.z >= upper_bound.z;
}

// Device only

template <typename Dim>
CUTLASS_DEVICE void maybe_mask_qk_tiles(
    Dim& q_shape,
    Dim orig_shape,
    Dim dilation,
    Dim dilation_idx);

template <>
CUTLASS_DEVICE void maybe_mask_qk_tiles(
    NA1dDim& q_shape,
    NA1dDim orig_shape,
    NA1dDim dilation,
    NA1dDim dilation_idx) {
  auto num_tiles_with_fewer_tokens = (q_shape * dilation) - orig_shape;
  // TODO: assert num_tiles_with_fewer_tokens >= 0
  if (dilation_idx.x >= dilation.x - num_tiles_with_fewer_tokens.x) {
    q_shape.x = q_shape.x - 1;
  }
}

template <>
CUTLASS_DEVICE void maybe_mask_qk_tiles(
    NA2dDim& q_shape,
    NA2dDim orig_shape,
    NA2dDim dilation,
    NA2dDim dilation_idx) {
  auto num_tiles_with_fewer_tokens = (q_shape * dilation) - orig_shape;
  // TODO: assert num_tiles_with_fewer_tokens >= 0
  if (dilation_idx.x >= dilation.x - num_tiles_with_fewer_tokens.x) {
    q_shape.x = q_shape.x - 1;
  }
  if (dilation_idx.y >= dilation.y - num_tiles_with_fewer_tokens.y) {
    q_shape.y = q_shape.y - 1;
  }
}

template <>
CUTLASS_DEVICE void maybe_mask_qk_tiles(
    NA3dDim& q_shape,
    NA3dDim orig_shape,
    NA3dDim dilation,
    NA3dDim dilation_idx) {
  auto num_tiles_with_fewer_tokens = (q_shape * dilation) - orig_shape;
  // TODO: assert num_tiles_with_fewer_tokens >= 0
  if (dilation_idx.x >= dilation.x - num_tiles_with_fewer_tokens.x) {
    q_shape.x = q_shape.x - 1;
  }
  if (dilation_idx.y >= dilation.y - num_tiles_with_fewer_tokens.y) {
    q_shape.y = q_shape.y - 1;
  }
  if (dilation_idx.z >= dilation.z - num_tiles_with_fewer_tokens.z) {
    q_shape.z = q_shape.z - 1;
  }
}

template <bool... IsCausalList>
struct CausalMask;

template <bool IsDim0Causal>
struct CausalMask<IsDim0Causal> {
  static constexpr int kRank = 1;
  static constexpr bool Dim0 = IsDim0Causal;

  static constexpr bool AnyCausalDims = IsDim0Causal;
};

template <bool IsDim0Causal, bool IsDim1Causal>
struct CausalMask<IsDim0Causal, IsDim1Causal> {
  static constexpr int kRank = 2;
  static constexpr bool Dim0 = IsDim0Causal;
  static constexpr bool Dim1 = IsDim1Causal;

  static constexpr bool AnyCausalDims = IsDim0Causal || IsDim1Causal;
};

template <bool IsDim0Causal, bool IsDim1Causal, bool IsDim2Causal>
struct CausalMask<IsDim0Causal, IsDim1Causal, IsDim2Causal> {
  static constexpr int kRank = 3;
  static constexpr bool Dim0 = IsDim0Causal;
  static constexpr bool Dim1 = IsDim1Causal;
  static constexpr bool Dim2 = IsDim2Causal;

  static constexpr bool AnyCausalDims =
      IsDim0Causal || IsDim1Causal || IsDim2Causal;
};

// No causal masking
template <int NADim>
struct GetNonCausalMask;

template <>
struct GetNonCausalMask<1> {
  using type = CausalMask<false>;
};

template <>
struct GetNonCausalMask<2> {
  using type = CausalMask<false, false>;
};

template <>
struct GetNonCausalMask<3> {
  using type = CausalMask<false, false, false>;
};

// NA Mask
template <bool DoCausalMasking_>
struct NeighborhoodAttentionMaskBase {
  static constexpr bool DoCausalMasking = DoCausalMasking_;
};

// Standard NA
template <>
struct NeighborhoodAttentionMaskBase<false> {
  static constexpr bool DoCausalMasking = false;

  int32_t window_size;
  int32_t window_radius;
  int32_t spatial_extent;
#ifdef _NATTEN_FNA_MASK_CACHE_LAST_START
  int32_t last_start;
#endif

  CUTLASS_DEVICE NeighborhoodAttentionMaskBase(
      int32_t kernel_size,
      int32_t spatial_extent_)
      : window_size(kernel_size),
        window_radius(kernel_size / 2),
#ifdef _NATTEN_FNA_MASK_CACHE_LAST_START
        spatial_extent(spatial_extent_),
        last_start(spatial_extent_ - window_radius - 1) {
  }
#else
        spatial_extent(spatial_extent_) {
  }
#endif

  CUTLASS_DEVICE int32_t get_window_start(int32_t index) const {
#ifdef _NATTEN_FNA_MASK_CACHE_LAST_START
    return cutlass::fast_max(index - window_radius, 0) +
        cutlass::fast_min(last_start - index, 0);
#else
    return cutlass::fast_max(index - window_radius, 0) +
        (index + window_radius >= spatial_extent) *
        (spatial_extent - index - window_radius - 1);
#endif
  }

  CUTLASS_DEVICE int32_t get_window_end(int32_t index, int32_t start) const {
    return start + window_size;
  }

  CUTLASS_DEVICE int32_t get_window_end_(int32_t index) const {
#ifdef _NATTEN_FNA_MASK_CACHE_LAST_START
    return cutlass::fast_max(index - window_radius, 0) +
        cutlass::fast_min(last_start - index, 0) + window_size;
#else
    return cutlass::fast_max(index - window_radius, 0) +
        (index + window_radius >= spatial_extent) *
        (spatial_extent - index - window_radius - 1) +
        window_size;
#endif
  }

  CUTLASS_DEVICE int32_t get_backward_window_start(int32_t index) const {
    return (index >= window_size) * (index - window_radius);
  }

  CUTLASS_DEVICE int32_t get_backward_window_end(int32_t index) const {
    return (index >= spatial_extent - window_size)
        ? (spatial_extent)
        : (index + (window_radius + 1));
  }

  CUTLASS_DEVICE int32_t get_rpb_start(int32_t index) const {
    return cutlass::fast_max(
        window_radius + cutlass::fast_max(window_radius - index, 0) -
            cutlass::fast_max(index - spatial_extent + window_radius + 1, 0),
        0);
  }
};

// Causal NA
template <>
struct NeighborhoodAttentionMaskBase<true> {
  static constexpr bool DoCausalMasking = true;

  int32_t window_size;
  int32_t window_radius;
  int32_t spatial_extent;

  CUTLASS_DEVICE NeighborhoodAttentionMaskBase(
      int32_t kernel_size,
      int32_t spatial_extent_)
      : window_size(kernel_size),
        window_radius(kernel_size / 2),
        spatial_extent(spatial_extent_) {}

  CUTLASS_DEVICE int32_t get_window_start(int32_t index) const {
    return cutlass::fast_max(index - window_size + 1, 0);
  }

  CUTLASS_DEVICE int32_t get_window_end(int32_t index, int32_t start) const {
    return cutlass::fast_min(index + 1, spatial_extent);
  }

  CUTLASS_DEVICE int32_t get_window_end_(int32_t index) const {
    return cutlass::fast_min(index + 1, spatial_extent);
  }

  CUTLASS_DEVICE int32_t get_backward_window_start(int32_t index) const {
    return index;
  }

  CUTLASS_DEVICE int32_t get_backward_window_end(int32_t index) const {
    return cutlass::fast_min(index + window_size, spatial_extent);
  }

  CUTLASS_DEVICE int32_t get_rpb_start(int32_t index) const {
    printf(
        "FATAL: RPB kernels do not support causal masking. Please open an issue.");
    asm volatile("brkpt;\n");
    return 0;
  }
};

// TODO: figure out an alternative to partial specializations for
// literally every dim. I'm pretty sure there's a better way of handling
// it with less repetition. It might require refactoring this whole
// header file, particularly how NA.dDim structs are defined.

CUTLASS_DEVICE int32_t align_down(int32_t tile, int32_t upper_bound) {
  return (tile / upper_bound) * upper_bound;
}

template <int NADim, typename CausalMask>
struct NeighborhoodAttentionMask;

template <int NADim, typename CausalMask_>
struct NeighborhoodAttentionMask {
  static_assert(NADim >= 1 && NADim < 4);
  static_assert(NADim == CausalMask_::kRank);

  using CausalMask = CausalMask_;
};

template <typename CausalMask_>
struct NeighborhoodAttentionMask<1, CausalMask_> {
  static constexpr int kRank = 1;

  using MaskType0 = NeighborhoodAttentionMaskBase<CausalMask_::Dim0>;
  using Dim = typename natten::cuda::fna::GetDim<kRank>::type;

  MaskType0 mask_0;

  CUTLASS_DEVICE NeighborhoodAttentionMask(Dim kernel_size, Dim spatial_extent_)
      : mask_0(kernel_size.x, spatial_extent_.x) {}

  CUTLASS_DEVICE Dim get_window_start(Dim index) const {
    return Dim(mask_0.get_window_start(index.x));
  }

  CUTLASS_DEVICE Dim get_window_end(Dim index, Dim start) const {
    return Dim(mask_0.get_window_end(index.x, start.x));
  }

  CUTLASS_DEVICE Dim get_window_end_(Dim index) const {
    return Dim(mask_0.get_window_end_(index.x));
  }

  CUTLASS_DEVICE Dim get_last_key_block_for_query_block(
      Dim index,
      Dim key_tile_shape,
      Dim last_possible_key_tile) const {
    return Dim(cutlass::fast_min(
        align_down(mask_0.get_window_end_(index.x) - 1, key_tile_shape.x),
        last_possible_key_tile.x));
  }

  CUTLASS_DEVICE Dim get_backward_window_start(Dim index) const {
    return Dim(mask_0.get_backward_window_start(index.x));
  }

  CUTLASS_DEVICE Dim get_backward_window_end(Dim index) const {
    return Dim(mask_0.get_backward_window_end(index.x));
  }

  CUTLASS_DEVICE Dim get_rpb_start(Dim index) const {
    return Dim(mask_0.get_rpb_start(index.x));
  }
};

template <typename CausalMask_>
struct NeighborhoodAttentionMask<2, CausalMask_> {
  static constexpr int kRank = 2;

  using MaskType0 = NeighborhoodAttentionMaskBase<CausalMask_::Dim0>;
  using MaskType1 = NeighborhoodAttentionMaskBase<CausalMask_::Dim1>;
  using Dim = typename natten::cuda::fna::GetDim<kRank>::type;

  MaskType0 mask_0;
  MaskType1 mask_1;

  CUTLASS_DEVICE NeighborhoodAttentionMask(Dim kernel_size, Dim spatial_extent_)
      : mask_0(kernel_size.x, spatial_extent_.x),
        mask_1(kernel_size.y, spatial_extent_.y) {}

  CUTLASS_DEVICE Dim get_window_start(Dim index) const {
    return Dim(
        mask_0.get_window_start(index.x), mask_1.get_window_start(index.y));
  }

  CUTLASS_DEVICE Dim get_window_end(Dim index, Dim start) const {
    return Dim(
        mask_0.get_window_end(index.x, start.x),
        mask_1.get_window_end(index.y, start.y));
  }

  CUTLASS_DEVICE Dim get_window_end_(Dim index) const {
    return Dim(
        mask_0.get_window_end_(index.x), mask_1.get_window_end_(index.y));
  }

  CUTLASS_DEVICE Dim get_last_key_block_for_query_block(
      Dim index,
      Dim key_tile_shape,
      Dim last_possible_key_tile) const {
    return Dim(
        cutlass::fast_min(
            align_down(mask_0.get_window_end_(index.x) - 1, key_tile_shape.x),
            last_possible_key_tile.x),
        cutlass::fast_min(
            align_down(mask_1.get_window_end_(index.y) - 1, key_tile_shape.y),
            last_possible_key_tile.y));
  }

  CUTLASS_DEVICE Dim get_backward_window_start(Dim index) const {
    return Dim(
        mask_0.get_backward_window_start(index.x),
        mask_1.get_backward_window_start(index.y));
  }

  CUTLASS_DEVICE Dim get_backward_window_end(Dim index) const {
    return Dim(
        mask_0.get_backward_window_end(index.x),
        mask_1.get_backward_window_end(index.y));
  }

  CUTLASS_DEVICE Dim get_rpb_start(Dim index) const {
    return Dim(mask_0.get_rpb_start(index.x), mask_1.get_rpb_start(index.y));
  }
};

template <typename CausalMask_>
struct NeighborhoodAttentionMask<3, CausalMask_> {
  static constexpr int kRank = 3;

  using MaskType0 = NeighborhoodAttentionMaskBase<CausalMask_::Dim0>;
  using MaskType1 = NeighborhoodAttentionMaskBase<CausalMask_::Dim1>;
  using MaskType2 = NeighborhoodAttentionMaskBase<CausalMask_::Dim2>;
  using Dim = typename natten::cuda::fna::GetDim<kRank>::type;

  MaskType0 mask_0;
  MaskType1 mask_1;
  MaskType2 mask_2;

  CUTLASS_DEVICE NeighborhoodAttentionMask(Dim kernel_size, Dim spatial_extent_)
      : mask_0(kernel_size.x, spatial_extent_.x),
        mask_1(kernel_size.y, spatial_extent_.y),
        mask_2(kernel_size.z, spatial_extent_.z) {}

  CUTLASS_DEVICE Dim get_window_start(Dim index) const {
    return Dim(
        mask_0.get_window_start(index.x),
        mask_1.get_window_start(index.y),
        mask_2.get_window_start(index.z));
  }

  CUTLASS_DEVICE Dim get_window_end(Dim index, Dim start) const {
    return Dim(
        mask_0.get_window_end(index.x, start.x),
        mask_1.get_window_end(index.y, start.y),
        mask_2.get_window_end(index.z, start.z));
  }

  CUTLASS_DEVICE Dim get_window_end_(Dim index) const {
    return Dim(
        mask_0.get_window_end_(index.x),
        mask_1.get_window_end_(index.y),
        mask_2.get_window_end_(index.z));
  }

  CUTLASS_DEVICE Dim get_last_key_block_for_query_block(
      Dim index,
      Dim key_tile_shape,
      Dim last_possible_key_tile) const {
    return Dim(
        cutlass::fast_min(
            align_down(mask_0.get_window_end_(index.x) - 1, key_tile_shape.x),
            last_possible_key_tile.x),
        cutlass::fast_min(
            align_down(mask_1.get_window_end_(index.y) - 1, key_tile_shape.y),
            last_possible_key_tile.y),
        cutlass::fast_min(
            align_down(mask_2.get_window_end_(index.z) - 1, key_tile_shape.z),
            last_possible_key_tile.z));
  }

  CUTLASS_DEVICE Dim get_backward_window_start(Dim index) const {
    return Dim(
        mask_0.get_backward_window_start(index.x),
        mask_1.get_backward_window_start(index.y),
        mask_2.get_backward_window_start(index.z));
  }

  CUTLASS_DEVICE Dim get_backward_window_end(Dim index) const {
    return Dim(
        mask_0.get_backward_window_end(index.x),
        mask_1.get_backward_window_end(index.y),
        mask_2.get_backward_window_end(index.z));
  }

  CUTLASS_DEVICE Dim get_rpb_start(Dim index) const {
    return Dim(
        mask_0.get_rpb_start(index.x),
        mask_1.get_rpb_start(index.y),
        mask_2.get_rpb_start(index.z));
  }
};

template <typename Dim>
CUTLASS_DEVICE void increment_tile(
    Dim& tile,
    const Dim& inc_value,
    const Dim& upper_bound);

template <>
CUTLASS_DEVICE void increment_tile(
    NA1dDim& tile,
    const NA1dDim& inc_value,
    const NA1dDim& upper_bound) {
  tile.x += inc_value.x;
}

template <>
CUTLASS_DEVICE void increment_tile(
    NA2dDim& tile,
    const NA2dDim& inc_value,
    const NA2dDim& upper_bound) {
  tile.y = tile.y + inc_value.y;
  tile.x = tile.x + (tile.y >= upper_bound.y) * inc_value.x;
  tile.y = (tile.y < upper_bound.y) * tile.y;
}

template <>
CUTLASS_DEVICE void increment_tile(
    NA3dDim& tile,
    const NA3dDim& inc_value,
    const NA3dDim& upper_bound) {
  tile.z = tile.z + inc_value.z;
  if (tile.z >= upper_bound.z) {
    tile.z = 0;
    tile.y += inc_value.y;
  }
  if (tile.y >= upper_bound.y) {
    tile.y = 0;
    tile.x += inc_value.x;
  }
}

template <typename Dim>
CUTLASS_DEVICE void increment_tile(
    Dim& tile,
    const Dim& inc_value,
    const Dim& lower_bound,
    const Dim& upper_bound);

template <>
CUTLASS_DEVICE void increment_tile(
    NA1dDim& tile,
    const NA1dDim& inc_value,
    const NA1dDim& lower_bound,
    const NA1dDim& upper_bound) {
  tile.x += inc_value.x;
}

template <>
CUTLASS_DEVICE void increment_tile(
    NA2dDim& tile,
    const NA2dDim& inc_value,
    const NA2dDim& lower_bound,
    const NA2dDim& upper_bound) {
  tile.y = tile.y + inc_value.y;
  if (tile.y >= upper_bound.y) {
    tile.y = lower_bound.y;
    tile.x += inc_value.x;
  }
}

template <>
CUTLASS_DEVICE void increment_tile(
    NA3dDim& tile,
    const NA3dDim& inc_value,
    const NA3dDim& lower_bound,
    const NA3dDim& upper_bound) {
  tile.z = tile.z + inc_value.z;
  if (tile.z >= upper_bound.z) {
    tile.z = lower_bound.z;
    tile.y += inc_value.y;
  }
  if (tile.y >= upper_bound.y) {
    tile.y = lower_bound.y;
    tile.x += inc_value.x;
  }
}

template <typename Dim>
CUTLASS_DEVICE Dim increment_tile_return(
    const Dim& tile,
    const Dim& inc_value,
    const Dim& lower_bound,
    const Dim& upper_bound);
template <typename Dim>
CUTLASS_DEVICE Dim increment_tile_return(
    const Dim& tile,
    const Dim& inc_value,
    const Dim& upper_bound);

template <>
CUTLASS_DEVICE NA1dDim increment_tile_return(
    const NA1dDim& tile,
    const NA1dDim& inc_value,
    const NA1dDim& lower_bound,
    const NA1dDim& upper_bound) {
  return NA1dDim(tile.x + inc_value.x);
}

template <>
CUTLASS_DEVICE NA1dDim increment_tile_return(
    const NA1dDim& tile,
    const NA1dDim& inc_value,
    const NA1dDim& upper_bound) {
  return NA1dDim(tile.x + inc_value.x);
}

template <>
CUTLASS_DEVICE NA2dDim increment_tile_return(
    const NA2dDim& tile,
    const NA2dDim& inc_value,
    const NA2dDim& lower_bound,
    const NA2dDim& upper_bound) {
  auto x = tile.x;
  auto y = tile.y + inc_value.y;
  if (y >= upper_bound.y) {
    y = lower_bound.y;
    x += inc_value.x;
  }
  return NA2dDim(x, y);
}

template <>
CUTLASS_DEVICE NA2dDim increment_tile_return(
    const NA2dDim& tile,
    const NA2dDim& inc_value,
    const NA2dDim& upper_bound) {
  auto x = tile.x;
  auto y = tile.y + inc_value.y;
  if (y >= upper_bound.y) {
    y = 0;
    x += inc_value.x;
  }
  return NA2dDim(x, y);
}

template <>
CUTLASS_DEVICE NA3dDim increment_tile_return(
    const NA3dDim& tile,
    const NA3dDim& inc_value,
    const NA3dDim& lower_bound,
    const NA3dDim& upper_bound) {
  auto x = tile.x;
  auto y = tile.y;
  auto z = tile.z + inc_value.z;
  if (z >= upper_bound.z) {
    z = lower_bound.z;
    y += inc_value.y;
  }
  if (y >= upper_bound.y) {
    y = lower_bound.y;
    x += inc_value.x;
  }
  return NA3dDim(x, y, z);
}

template <>
CUTLASS_DEVICE NA3dDim increment_tile_return(
    const NA3dDim& tile,
    const NA3dDim& inc_value,
    const NA3dDim& upper_bound) {
  auto x = tile.x;
  auto y = tile.y;
  auto z = tile.z + inc_value.z;
  if (z >= upper_bound.z) {
    z = 0;
    y += inc_value.y;
  }
  if (y >= upper_bound.y) {
    y = 0;
    x += inc_value.x;
  }
  return NA3dDim(x, y, z);
}

template <typename Dim>
CUTLASS_DEVICE bool is_tile_last(Dim tile, Dim inc_value, Dim upper_bound);

template <>
CUTLASS_DEVICE bool is_tile_last(
    NA1dDim tile,
    NA1dDim inc_value,
    NA1dDim upper_bound) {
  return tile.x + inc_value.x >= upper_bound.x;
}

template <>
CUTLASS_DEVICE bool is_tile_last(
    NA2dDim tile,
    NA2dDim inc_value,
    NA2dDim upper_bound) {
  return tile.y + inc_value.y >= upper_bound.y &&
      tile.x + inc_value.x >= upper_bound.x;
}

template <>
CUTLASS_DEVICE bool is_tile_last(
    NA3dDim tile,
    NA3dDim inc_value,
    NA3dDim upper_bound) {
  return tile.z + inc_value.z >= upper_bound.z &&
      tile.y + inc_value.y >= upper_bound.y &&
      tile.x + inc_value.x >= upper_bound.x;
}

template <typename Dim>
CUTLASS_DEVICE Dim count_tiles(Dim first, Dim last, Dim tile_shape);

template <>
CUTLASS_DEVICE NA1dDim
count_tiles(NA1dDim first, NA1dDim last, NA1dDim tile_shape) {
  return NA1dDim((last.x / tile_shape.x) - (first.x / tile_shape.x) + 1);
}

template <>
CUTLASS_DEVICE NA2dDim
count_tiles(NA2dDim first, NA2dDim last, NA2dDim tile_shape) {
  return NA2dDim(
      ((last.x / tile_shape.x) - (first.x / tile_shape.x) + 1),
      ((last.y / tile_shape.y) - (first.y / tile_shape.y) + 1));
}

template <>
CUTLASS_DEVICE NA3dDim
count_tiles(NA3dDim first, NA3dDim last, NA3dDim tile_shape) {
  return NA3dDim(
      ((last.x / tile_shape.x) - (first.x / tile_shape.x) + 1),
      ((last.y / tile_shape.y) - (first.y / tile_shape.y) + 1),
      ((last.z / tile_shape.z) - (first.z / tile_shape.z) + 1));
}

} // namespace fna
} // namespace cuda
} // namespace natten
