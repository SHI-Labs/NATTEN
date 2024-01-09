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
    \brief Holds common macros for tiled implementations.
*/

#pragma once

#define KERNEL_SIZE_13 13
#define KERNEL_SIZE_11 11
#define KERNEL_SIZE_9 9
#define KERNEL_SIZE_7 7
#define KERNEL_SIZE_5 5
#define KERNEL_SIZE_3 3
#define NEIGHBORHOOD_SIZE_13 6
#define NEIGHBORHOOD_SIZE_11 5
#define NEIGHBORHOOD_SIZE_9 4
#define NEIGHBORHOOD_SIZE_7 3
#define NEIGHBORHOOD_SIZE_5 2
#define NEIGHBORHOOD_SIZE_3 1
// Always keep batchthreads 1, because we want each thread block to process one
// 1 sample 1 head
#define BATCHTHREADS_13 1
#define BATCHTHREADS_11 1
#define BATCHTHREADS_9 1
#define BATCHTHREADS_7 1
#define BATCHTHREADS_5 1
#define BATCHTHREADS_3 1
// Tile is the number of pixels across each axis that are processed within a
// single threadblock So far the best tile size for Kernel size 7 is 3x3.
#define TILE_9 3
#define TILE_7 3
#define TILE_5 4
#define TILE_3 7

#define TILE_11_X 2
#define TILE_11_Y 3
#define TILE_13_X 2
#define TILE_13_Y 3
// Each of the 3x3 pixels has 7x7 key neighbors in this case, therefore the tile
// size for keys will 7 + 3 - 1 = 9x9
#define KTILE_9 11
#define KTILE_7 9
#define KTILE_5 8
#define KTILE_3 9

#define KTILE_11_X 12
#define KTILE_11_Y 13
#define KTILE_13_X 14
#define KTILE_13_Y 15
// 7x7 kernel, and we want each threadblock to process the entire neighborhood
// for each QUERY in its tile, so we'll have 7x7 * 3x3 = 21x21 Also keep in mind
// these 21 threads are across each axis, so it's 21x21 threads total 21x21 =
// 441 < 1024 Ensure it's less than 1024, which is the max number of threads per
// threadblock
#define XYTHREADS_9 27
#define XYTHREADS_7 21
#define XYTHREADS_5 20
#define XYTHREADS_3 21

#define XTHREADS_11 33
#define YTHREADS_11 22
#define XTHREADS_13 39
#define YTHREADS_13 26

// DIM is fixed at 32 for now
#define DIM_32 32
#define DIMHALF_32 16 // FP16 stored in half2 => half the dims
// There's 32 * 3x3 QUERY cells to store, and 32 * 10x10 KEY cells
// The former is 288 < 441 threads, so each thread can copy over one QUERY cell
// exactly, and we'll have empty threads too But that's not the case for the
// latter, which is 3200 and it's not < 441 But we can have each thread load
// more cells instead. 8 is optimal since it will maximize utility So copy 8
// dims per KEY pixel in each thread
#define KITERS_32 8
#define KHALFITERS_32 4 // FP16 stored in half2 => half the dims
// and DIM = 32 / 8 = 4, hence 4 is the stride.
#define KSTRIDE_32 4
// For kernel size 5, we have to do 2 query dims per thread, because we have
// fewer threads in each threadblock than the total number of queries. For
// kernel size 3, we have to read 2 query dims per thread
#define QITERS_5 2
#define QSTRIDE_5 16
#define QITERS_3 4
#define QSTRIDE_3 8
#define QITERS_3_HALF 2
#define QSTRIDE_3_HALF 8
