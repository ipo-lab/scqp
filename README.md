# scqp
SCQP is a first-order splitting method for convex quadratic programs. The QP solver is implemented in numpy (for dense QPs) and scipy (for sparse QPs) and invokes a basic implementation of the ADMM algorithm.

For more information please see our publication:

[arXiv (preprint)](https://github.com/ipo-lab/scqpth)

For experimental results please see [scqp_bench](https://github.com/ipo-lab/scqp_bench)

## Core Dependencies:
To use the ADMM solver you will need to install [numpy](https://numpy.org),  [Scipy](https://scipy.org) and [qdldl](https://github.com/osqp/qdldl-python).
Please see requirements.txt for full build details.
