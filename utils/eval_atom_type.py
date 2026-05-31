# =============================================================================
# From: https://github.com/guanjq/targetdiff
#
# MIT License
#
# Copyright (c) 2023 Jiaqi Guan
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
# =============================================================================

from collections import Counter
from scipy import spatial as sci_spatial
import numpy as np

# ATOM_TYPE_DISTRIBUTION = {
#     6: 1585004,
#     7: 276248,
#     8: 400236,
#     9: 30871,
#     15: 26288,
#     16: 26529,
#     17: 15210,
# }

ATOM_TYPE_DISTRIBUTION = {
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}


def eval_atom_type_distribution(pred_counter: Counter):
    total_num_atoms = sum(pred_counter.values())
    pred_atom_distribution = {}
    for k in ATOM_TYPE_DISTRIBUTION:
        pred_atom_distribution[k] = pred_counter[k] / total_num_atoms
    print('pred atom distribution: ', pred_atom_distribution)
    print('ref  atom distribution: ', ATOM_TYPE_DISTRIBUTION)
    js = sci_spatial.distance.jensenshannon(np.array(list(ATOM_TYPE_DISTRIBUTION.values())),
                                            np.array(list(pred_atom_distribution.values())))
    return js
