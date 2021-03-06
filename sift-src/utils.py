#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Contains a class for creating a plan, allocating arrays, compiling kernels and other things like that
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "BSD"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-06-13"
__status__ = "beta"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

from math import ceil
import numpy

def calc_size(shape, blocksize):
    """
    Calculate the optimal size for a kernel according to the workgroup size
    """
    if "__len__" in dir(blocksize):
        return tuple((i + j - 1) & ~(j - 1) for i, j in zip(shape, blocksize))
    else:
        return tuple((i + blocksize - 1) & ~(blocksize - 1) for i in shape)


def kernel_size(sigma, odd=False, cutoff=4):
    """
    Calculate the optimal kernel size for a convolution with sigma
    
    @param sigma: width of the gaussian 
    @param odd: enforce the kernel to be odd (more precise ?)
    """
    size = int(ceil(2 * cutoff * sigma + 1))
    if odd and size % 2 == 0:
        size += 1
    return size

def sizeof(shape, dtype="uint8"):
    """
    Calculate the number of bytes needed to allocate for a given structure
    
    @param shape: size or tuple of sizes
    @param dtype: data type
    """
    itemsize = numpy.dtype(dtype).itemsize
    cnt = 1
    if "__len__" in dir(shape):
        for dim in shape:
            cnt *= dim
    else:
        cnt = int(shape)
    return cnt * itemsize
