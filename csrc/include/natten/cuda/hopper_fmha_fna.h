/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
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

#pragma once

namespace natten {
namespace cuda {
namespace hopper {

enum class HopperKernelSchedule {
  NonPersistent,
  WSCooperative,
  WSPingpong,
  Invalid
};

inline auto kernel_type_int_to_enum_type(int kernel_type) {
  switch (kernel_type) {
    case 0:
      return HopperKernelSchedule::NonPersistent;
    case 1:
      return HopperKernelSchedule::WSCooperative;
    case 2:
      return HopperKernelSchedule::WSPingpong;
  }
  std::cerr
      << "Invalid value for argument kernel_type; expected either 0, 1, or 2, got "
      << kernel_type << ".\n";
  return HopperKernelSchedule::Invalid;
}

} // namespace hopper
} // namespace cuda
} // namespace natten
