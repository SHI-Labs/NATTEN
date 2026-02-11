#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################


from typing import Dict, List

from ....types import QKTileShapeType

__all__ = ["get_flash_fmha_bwd_configs"]


_FLASH_FMHA_BACKWARD_CONFIGS = {
    1: {
        80: {
            32: [
                ((128,), (128,)),
            ],
            64: [
                ((128,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (128,)),
            ],
            192: [
                ((64,), (80,)),
            ],
            256: [
                ((64,), (64,)),
            ],
        },
        86: {
            32: [
                ((64,), (128,)),
            ],
            64: [
                ((64,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (96,)),
            ],
            192: [
                ((64,), (64,)),
            ],
            256: [
                ((32,), (64,)),
            ],
        },
        89: {
            32: [
                ((64,), (128,)),
            ],
            64: [
                ((64,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (96,)),
            ],
            192: [
                ((64,), (64,)),
            ],
            256: [
                ((32,), (64,)),
            ],
        },
    },
}

def get_flash_fmha_bwd_configs():
    return _FLASH_FMHA_BACKWARD_CONFIGS
