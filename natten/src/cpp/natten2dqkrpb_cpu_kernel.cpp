/*
NATTEN2D-QKRPB TORCH EXTENSION (CPU)

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
void natten2dqkrpb_cpu_forward_kernel(
    const at::TensorAccessor<scalar_t, 5> query,
    const at::TensorAccessor<scalar_t, 5> key,
    const at::TensorAccessor<scalar_t, 3> rpb,
    at::TensorAccessor<scalar_t, 5> attn,
    const int height, 
    const int width,
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
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
        const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
        int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3);
        scalar_t* _qaddr = query.data() + queryOffset;
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            Vec updt = Vec(scalar_t(0));
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            scalar_t* _kaddr = key.data() + keyOffset;
            int64_t d1 = 0;
            for (; d1 < dim - (dim % Vec::size()); d1+=Vec::size())
                updt = at::vec::fmadd(Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
            for (; d1 < dim; ++d1)
                updt[d1] += _qaddr[d1] * _kaddr[d1];
            const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
            attn.data()[index] = rpb.data()[rpbIndex] + at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
            index++;
        }}
    }});
#else
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            scalar_t updt = scalar_t(0);
            const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
            const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
            const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
            const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
            updt += rpb.data()[rpbIndex];
            attn.data()[index] = updt;
        }}
    }});
#endif
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten2dq_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 5> d_query,
    const at::TensorAccessor<scalar_t, 5> d_attn,
    const at::TensorAccessor<scalar_t, 5> key,
    const int height, 
    const int width,
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
            for (int i = 0; i < height; i++) {
            const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            for (int j = 0; j < width; j++) {
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    scalar_t d_query_update = scalar_t(0);
                    int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3);
                    const int keyOffset = b * key.stride(0) + h * key.stride(1) + d;
                    for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
                    for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
                        const int keyIndex = keyOffset + xi * key.stride(2) + xj * key.stride(3);
                        d_query_update += d_attn.data()[attnOffset] * key.data()[keyIndex];
                        ++attnOffset;
                    }}
                    const int linearIndex = b*d_query.stride(0) + h*d_query.stride(1) + i*d_query.stride(2) + j*d_query.stride(3) + d;
                    d_query.data()[linearIndex] = d_query_update;
                }
            }}
        }});
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten2drpb_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 3> d_rpb,
    const at::TensorAccessor<scalar_t, 5> d_attn,
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
    for (int h = start; h < end; h++) {
        for (int i = 0; i < height; i++) {
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        for (int j = 0; j < width; j++) {
            const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
            for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                scalar_t d_rpb_update = scalar_t(0);
                int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + ki*KERNEL_SIZE+kj;
                for (int b=0; b < batch_size; ++b){
                    d_rpb_update += d_attn.data()[attnOffset];
                    attnOffset += d_attn.stride(0);
                }
                const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1) + (pj+kj) * d_rpb.stride(2);
                d_rpb.data()[index] += d_rpb_update;
            }}
        }}
    }});
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten2dk_cpu_backward_kernel(
    at::TensorAccessor<scalar_t, 5> d_key,
    const at::TensorAccessor<scalar_t, 5> d_attn,
    const at::TensorAccessor<scalar_t, 5> query,
    const int height, 
    const int width,
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
            for (int i = 0; i < height; i++) {
            const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int ei = get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            for (int j = 0; j < width; j++) {
                const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int ej = get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
                    const int queryOffset = b * query.stride(0) + h * query.stride(1) + d;
                    scalar_t d_key_update = scalar_t(0);
                    for (int xi=ni; xi < ei; xi+=dilation){
                    const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                    for (int xj=nj; xj < ej; xj+=dilation){
                        const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                        const int queryIndex = queryOffset + xi * query.stride(2) + xj * query.stride(3);
                        const int attnIndex = attnOffset + xi * d_attn.stride(2) + xj * d_attn.stride(3) + int((i-oni)/dilation)*KERNEL_SIZE + int((j-onj)/dilation);
                        d_key_update += query.data()[queryIndex] * d_attn.data()[attnIndex];
                    }}
                    const int linearIndex = b*d_key.stride(0) + h*d_key.stride(1) + i*d_key.stride(2) + j*d_key.stride(3) + d;
                    d_key.data()[linearIndex] = d_key_update;
                }
            }}
        }});
    }
}

torch::Tensor natten2dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int height = query.size(2);
    int width = query.size(3);
    int dim = query.size(4);
    int RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    CHECK_FEATMAP(height, width, kernel_size, dilation);

    auto attn = torch::zeros(
            {batch_size, heads, height, width, kernel_size*kernel_size}, query.options());

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten2dqkrpb_cpu_forward", ([&] {
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dqkrpb_cpu_forward_kernel, 
                query.accessor<scalar_t, 5>(), key.accessor<scalar_t, 5>(), 
                rpb.accessor<scalar_t, 3>(), attn.accessor<scalar_t, 5>(), 
                height, width, heads, kernel_size, dilation, dim, batch_size);
    }));
    return attn;
}

std::vector<torch::Tensor> natten2dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int height = query.size(2);
    int width = query.size(3);
    int dim = query.size(4);
    int kernel_size_sq = d_attn.size(4);
    int kernel_size = std::sqrt(kernel_size_sq);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    int RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX}, d_attn.options());

    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten2dqkrpb_backward_cpu", ([&] {
        const auto d_attn_a = d_attn.accessor<scalar_t, 5>();
        const auto query_a = query.accessor<scalar_t, 5>();
        const auto key_a = key.accessor<scalar_t, 5>();
        auto d_query_a = d_query.accessor<scalar_t, 5>();
        auto d_key_a = d_key.accessor<scalar_t, 5>();
        auto d_rpb_a = d_rpb.accessor<scalar_t, 3>();
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2drpb_cpu_backward_kernel, 
                d_rpb_a, d_attn_a, height, width, heads, kernel_size, dilation, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dq_cpu_backward_kernel, 
                d_query_a, d_attn_a, key_a, height, width, heads, kernel_size, dilation, dim, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dk_cpu_backward_kernel, 
                d_key_a, d_attn_a, query_a, height, width, heads, kernel_size, dilation, dim, batch_size);
    }));
    return {d_query, d_key, d_rpb};
}
} // namespace natten
