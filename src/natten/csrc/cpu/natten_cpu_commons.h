/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

inline int get_backward_window_start(const int index, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    return (index < KERNEL_SIZE * dilation) ? (index % dilation) : index - NEIGHBORHOOD_SIZE * dilation;
}


inline int get_backward_window_end(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    return (index >= length - KERNEL_SIZE * dilation) ? (length) : (index + (NEIGHBORHOOD_SIZE + 1) * dilation);
}


inline int get_window_start(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    if (dilation <= 1)
        return  std::max(index - NEIGHBORHOOD_SIZE, 0) + (index + NEIGHBORHOOD_SIZE >= length) * (length - index - NEIGHBORHOOD_SIZE - 1);
    int ni = index - NEIGHBORHOOD_SIZE * dilation;
    if (ni < 0)
        return index % dilation;
    if (index + NEIGHBORHOOD_SIZE * dilation >= length){
        const int imodd = index % dilation;
        const int a = int(length / dilation) * dilation;
        const int b = length - a;
        if (imodd < b)
            return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
        return a + imodd - KERNEL_SIZE * dilation;
    }
    return ni;
}


inline int get_pb_start(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    if (dilation <= 1)
        return NEIGHBORHOOD_SIZE + (index < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - index) + (index + NEIGHBORHOOD_SIZE >= length) * (length - index - 1 - NEIGHBORHOOD_SIZE);
    if (index - NEIGHBORHOOD_SIZE * dilation < 0)
        return KERNEL_SIZE - 1 - (index / dilation);
    if (index + NEIGHBORHOOD_SIZE * dilation >= length)
        return (length - index - 1) / dilation;
    return NEIGHBORHOOD_SIZE;
}

#define CHECK_SEQUENCE(length, kernel_size, dilation) TORCH_CHECK(length >= kernel_size*dilation, "Input sequence length must be greater than or equal to kernel size x dilation.")
#define CHECK_FEATMAP(height, width, kernel_size, dilation) TORCH_CHECK(height >= kernel_size*dilation && width >= kernel_size*dilation, "Input resolution must be greater than or equal to kernel size x dilation.")


#define _IN_LAUNCH_DNA_KNS(KS, NS, dilation, NAME, ...)                     \
({                                                                                                   \
    switch (dilation) {                                                                              \
        case 1:                                                                                      \
            NAME<KS, NS, 1, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 2:                                                                                      \
            NAME<KS, NS, 2, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 3:                                                                                      \
            NAME<KS, NS, 3, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 4:                                                                                      \
            NAME<KS, NS, 4, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 5:                                                                                      \
            NAME<KS, NS, 5, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 6:                                                                                      \
            NAME<KS, NS, 6, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 7:                                                                                      \
            NAME<KS, NS, 7, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 8:                                                                                      \
            NAME<KS, NS, 8, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 9:                                                                                      \
            NAME<KS, NS, 9, scalar_t>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 10:                                                                                     \
            NAME<KS, NS, 10, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 11:                                                                                     \
            NAME<KS, NS, 11, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 12:                                                                                     \
            NAME<KS, NS, 12, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 13:                                                                                     \
            NAME<KS, NS, 13, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 14:                                                                                     \
            NAME<KS, NS, 14, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 15:                                                                                     \
            NAME<KS, NS, 15, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 16:                                                                                     \
            NAME<KS, NS, 16, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
        default:                                                                                     \
            NAME<KS, NS, -1, scalar_t>(__VA_ARGS__);                    \
            break;                                                                                   \
    }                                                                                                \
})

#define LAUNCH_DNA_KNS(kernel_size, dilation, NAME, ...)                 \
({                                                                                                   \
    switch (kernel_size) {                                                                           \
        case 3:                                                                                      \
            _IN_LAUNCH_DNA_KNS(3, 1, dilation, NAME, __VA_ARGS__);          \
            break;                                                                                   \
        case 5:                                                                                      \
            _IN_LAUNCH_DNA_KNS(5, 2, dilation, NAME, __VA_ARGS__);          \
            break;                                                                                   \
        case 7:                                                                                      \
            _IN_LAUNCH_DNA_KNS(7, 3, dilation, NAME, __VA_ARGS__);          \
            break;                                                                                   \
        case 9:                                                                                      \
            _IN_LAUNCH_DNA_KNS(9, 4, dilation, NAME, __VA_ARGS__);          \
            break;                                                                                   \
        case 11:                                                                                     \
            _IN_LAUNCH_DNA_KNS(11, 5, dilation, NAME, __VA_ARGS__);         \
            break;                                                                                   \
        case 13:                                                                                     \
            _IN_LAUNCH_DNA_KNS(13, 6, dilation, NAME, __VA_ARGS__);         \
            break;                                                                                   \
        default:                                                                                     \
            _IN_LAUNCH_DNA_KNS(-1, -1, dilation, NAME, __VA_ARGS__);        \
            break;                                                                                   \
    }                                                                                                \
})
