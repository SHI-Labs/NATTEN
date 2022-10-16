/*
NATTEN1D-QKRPB TORCH EXTENSION (CPU)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif
#include "natten_cpu_commons.h"

namespace natten {

#define GRAIN_SIZE 0

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1dqkrpb_cpu_forward_kernel(
    const at::TensorAccessor<scalar_t, 4> query,
    const at::TensorAccessor<scalar_t, 4> key,
    const at::TensorAccessor<scalar_t, 2> rpb,
    at::TensorAccessor<scalar_t, 4> attn,
    const int length, 
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*length, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/length;
        const int i = x - indtmp1 * length;
        int indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
        const int queryOffset = batchHeadOffset + i * query.stride(2);
        int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        scalar_t* _qaddr = query.data() + queryOffset;
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
            Vec updt = Vec(scalar_t(0));
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2);
            scalar_t* _kaddr = key.data() + keyOffset;
            int64_t d1 = 0;
            for (; d1 < dim - (dim % Vec::size()); d1+=Vec::size())
                updt = at::vec::fmadd(Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
            for (; d1 < dim; ++d1)
                updt[d1] += _qaddr[d1] * _kaddr[d1];
            const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1);
            attn.data()[index] = rpb.data()[rpbIndex] + at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
            index++;
        }
    }});
#else
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                    scalar_t updt = scalar_t(0);
                    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                    const int queryOffset = batchHeadOffset + i * query.stride(2);
                    const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2);
                    for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                        updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                    const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki;
                    const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1);
                    updt += rpb.data()[rpbIndex];
                    attn.data()[index] = updt;
                }
            }
        }});
    }
#endif
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1dq_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 4> d_query,
    const at::TensorAccessor<scalar_t, 4> d_attn,
    const at::TensorAccessor<scalar_t, 4> key,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    scalar_t d_query_update = scalar_t(0);
                    int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2);
                    const int keyOffset = b * key.stride(0) + h * key.stride(1) + d;
                    for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
                        const int keyIndex = keyOffset + xi * key.stride(2);
                        d_query_update += d_attn.data()[attnOffset] * key.data()[keyIndex];
                        ++attnOffset;
                    }
                    const int linearIndex = b*d_query.stride(0) + h*d_query.stride(1) + i*d_query.stride(2) + d;
                    d_query.data()[linearIndex] = d_query_update;
                }
            }
        }});
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1drpb_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 2> d_rpb,
    const at::TensorAccessor<scalar_t, 4> d_attn,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
    for (int h = start; h < end; h++) {
        for (int i = 0; i < length; i++) {
            const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                scalar_t d_rpb_update = scalar_t(0);
                int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
                for (int b=0; b < batch_size; ++b){
                    d_rpb_update += d_attn.data()[attnOffset];
                    attnOffset += d_attn.stride(0);
                }
                const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1);
                d_rpb.data()[index] += d_rpb_update;
            }
        }
    }});
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1dk_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 4> d_key,
    const at::TensorAccessor<scalar_t, 4> d_attn,
    const at::TensorAccessor<scalar_t, 4> query,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int ei = get_backward_window_end(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
                    const int queryOffset = b * query.stride(0) + h * query.stride(1) + d;
                    scalar_t d_key_update = scalar_t(0);
                    for (int xi=ni; xi < ei; xi+=dilation){
                        const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                        const int queryIndex = queryOffset + xi * query.stride(2);
                        const int attnIndex = attnOffset + xi * d_attn.stride(2) + int((i-oni)/dilation);
                        d_key_update += query.data()[queryIndex] * d_attn.data()[attnIndex];
                    }
                    const int linearIndex = b*d_key.stride(0) + h*d_key.stride(1) + i*d_key.stride(2) + d;
                    d_key.data()[linearIndex] = d_key_update;
                }
            }
        }});
    }
}

torch::Tensor natten1dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int length = query.size(2);
    int dim = query.size(3);
    int RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    CHECK_SEQUENCE(length, kernel_size, dilation);

    auto attn = torch::zeros(
            {batch_size, heads, length, kernel_size}, query.options());

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten1dqkrpb_cpu_forward", ([&] {
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1dqkrpb_cpu_forward_kernel, 
                query.accessor<scalar_t, 4>(), key.accessor<scalar_t, 4>(), 
                rpb.accessor<scalar_t, 2>(), attn.accessor<scalar_t, 4>(), 
                length, heads, kernel_size, dilation, dim, batch_size);
    }));
    return attn;
}

std::vector<torch::Tensor> natten1dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int length = query.size(2);
    int dim = query.size(3);
    int kernel_size = d_attn.size(3);
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros({heads, RPB_MAX}, d_attn.options());

    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten1dqkrpb_backward_cpu", ([&] {
        const auto d_attn_a = d_attn.accessor<scalar_t, 4>();
        const auto query_a = query.accessor<scalar_t, 4>();
        const auto key_a = key.accessor<scalar_t, 4>();
        auto d_query_a = d_query.accessor<scalar_t, 4>();
        auto d_key_a = d_key.accessor<scalar_t, 4>();
        auto d_rpb_a = d_rpb.accessor<scalar_t, 2>();
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1drpb_cpu_backward_kernel, 
                d_rpb_a, d_attn_a, length, heads, kernel_size, dilation, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1dq_cpu_backward_kernel, 
                d_query_a, d_attn_a, key_a, length, heads, kernel_size, dilation, dim, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1dk_cpu_backward_kernel, 
                d_key_a, d_attn_a, query_a, length, heads, kernel_size, dilation, dim, batch_size);
    }));
    return {d_query, d_key, d_rpb};
}
} // namespace natten
