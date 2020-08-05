# factorization

## Contents
* [LU](https://github.com/joaopaulq/factorization/blob/master/src/qr.lu): L is lower triangular with 1's on the diagonal. U is upper triangular with pivots on the diagonal. No row exchanges as Gaussian elimination reduces square A to U.
* [QR](https://github.com/joaopaulq/factorization/blob/master/src/qr.py): Q has orthonormal columns, R is upper triangular. A has independent columns. Those are orthogonalized in Q by the Gram-Schmidt or Householder process. If A is square then Q⁻¹ = Q.T 

## Resources
* [MIT 18.06 - Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
* [Matrix Factorizations](http://math.mit.edu/~gs/linearalgebra/linearalgebra5_Matrix.pdf)
