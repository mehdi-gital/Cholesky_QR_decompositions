// Blocked Cholesky decomposition
#include "tnt.h"
using namespace TNT;
using namespace std;

template <class T>  // test if two matrices are equal
bool exactlyEqual(Array2D<T> A, Array2D<T> B) {
  int m = A.dim1();
  int n = A.dim2();

  int r = B.dim1();
  int c = B.dim2();

  bool v = true;

  if (m == r && n == c) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        v = (B[i][j] == A[i][j]);
        if (v == false) {
          return v;
        }
      }
    }

  } else {
    cout << "Not even the same dimensions!" << endl;
    return false;
  }
  return v;
}

template <class T>  // just transpose
Array2D<T> transpose(Array2D<T> A) {
  int m = A.dim1();
  int n = A.dim2();
  Array2D<T> B(n, m);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) B[j][i] = A[i][j];

  return B;
}

// template <class T>
Array2D<double> randMat(int n) {  // create a random upper triangular nxn matrix
  Array2D<double> M(n, n);

  srand((int)time(0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j < i) {
        M[i][j] = 0.;
      } else {
        M[i][j] = rand() % 5 + 1;
      }
    }
  }

  return M;
}

/////////// Cholesky 3 (simple loops - zeros copied at the end)

template <class T>
Array2D<T> chol3(const Array2D<T>& A) {
  const int m = A.dim1();
  const int n = A.dim2();

  if (m != n) {
    throw runtime_error("The matrix is not square!");
  }

  Array2D<T> ch(n, n);  // the Cholesky output

 ch = A.copy();

  for (int i = 0; i < n; i++) {  // the main loop

    ch[i][i] = sqrt(ch[i][i]);

    for (int j = i + 1; j < n; j++) {  // the inner loop for the hook
      ch[i][j] = ch[i][j] / ch[i][i];
      }

    for (int a = i + 1; a < n; a++) {
      for (int b = i + 1; b < n; b++) {
        ch[a][b] = ch[a][b] - ch[i][a] * ch[i][b];
      }
    }

  }

  for (int i = 0;i<n;i++){
      for (int j=0;j<i;j++){
          ch[i][j] = 0.;
      }
  }

  return ch;
}

////////////////////

template <class T> // The Frobenius norm
T frob(Array2D<T> A){
  const int m = A.dim1();
  const int n = A.dim2();

  T f;

  for (int i=0;i<m;i++){
    for (int j=0;j<n;j++){
      f += A[i][j]*A[i][j];
    }
  }
  return sqrt(f);
}

////////////// The inverse of a lower triangular matrix

template <class T> 
Array2D<T> lower_tri_inverse(const Array2D<T>& A) { // A must be lower triangular
  const int n = A.dim2();
  Array2D<T> B(n,n);

  for (int i=0;i<n;i++){
    B[i][i]=1./A[i][i]; // the diagonal entries
  }

  for (int i=1;i<n;i++){ // compute the lower triangle in B row by row
    for (int j=0;j<i;j++){
      for (int t=j;t<i;t++){
        B[i][j] += A[i][t]*B[t][j];
      }
      B[i][j] = (-1./A[i][i])*B[i][j];
    }
  }

  for (int i=0;i<n;i++){ // zeros for the upper triangle
    for (int j=i+1;j<n;j++){
      B[i][j] = 0.;
    }
  }

  return B;
}



////////////// Blocked Cholesky Test

template <class T>
Array2D<T> bCholTest(const Array2D<T>& A) {
  const int m = A.dim1();
  const int n = A.dim2();
  const int d = 5;

  if (m != n) {
    throw runtime_error("The matrix is not square!");
  }

  Array2D<T> ch(n, n);  // the output

  Array2D<T> U(d, d);              // the top left dxd minor
  Array2D<T> tempM(n - d, n - d);  // temp matrix

  const int q = n / d;  // the number of block operations
  const int r = n % d;  // the remaining bit

  ch = A.copy();  // overwrite everything on the output

  T x = 0.;  // temp variable

  int s;  // the shift

  for (int p = 0; p < q; p++) {  ////////////// THE BIG LOOP ////////////////

    s = p * d;

    for (int i = 0; i < d; i++) {  // extract the top-right minor
      for (int j = 0; j < d; j++) {
        U[i][j] = ch[i + s][j + s];  // s gives the correct shift
      }
    }

    U = chol3(U);  // compute a simple unblocked Cholesky: upper triangular U_{1,1}

    for (int i = 0; i < d;i++) {  // copy the top-left triangle back into the output
      for (int j = i; j < d; j++) {
        ch[i + s][j + s] = U[i][j];
      }
    }

    U = lower_tri_inverse(transpose(U));  // the inverse of the transpose of U_{1,1}

    for (int i = 0; i < d; i++) {  // compute the top-right triangle: U_{1,2}
                                   // which is dx(n-(p+1)*d)
      for (int j = 0; j < n - (p + 1) * d; j++) {
        x = 0.;
        for (int t = 0; t < d; t++) {
          x += U[i][t] * ch[t + s][j + s + d];  // Tricky shifts in (U_{1,2}^T)^(-1)*A_{1,2}
        }
        tempM[i][j] = x;
      }
    }

    // copy entries into the local top-right part of the output
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < n - (p + 1) * d; j++) {
        ch[i + s][j + s + d] = tempM[i][j];
      }
    }

    for (int i = 0; i < n - s - d;
         i++) {  // Working on the bottom-right triangle of size (n-(p+1)*d)^2
      for (int j = 0; j < n - s - d; j++) {
        x = 0.;
        for (int t = 0; t < d; t++) {
          x += ch[t + s][i + s + d] *
               ch[t + s][j + s + d];  // an entry of U_{1,2}^T*U_{1,2}
        }
        tempM[i][j] = x;  // tempM is U_{1,2}^T*U_{1,2}
        // ch[i+s+d][j+s+d] -= x;
      }
    }

    for (int i = 0; i < n - s - d;
         i++) {  // updating the bottom-right triangle of size (n-(p+1)*d)
      for (int j = 0; j < n - s - d; j++) {
        ch[i + s + d][j + s + d] -=
            tempM[i][j];  // now ch is locally = A_{2,2}-U_{1,2}^T*U_{1,2}
      }
    }
  }

  if (r != 0) {  // The last small block
    Array2D<T> V(r,r);

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < r; j++) {
        V[i][j] = ch[i + d * q][j + d * q];
      }
    }
    V = chol3(V);

    for (int i = 0; i < r; i++) {
      for (int j = i; j < r; j++) {
        ch[i + d * q][j + d * q] = V[i][j];
      }
    }
  }

  for (int i = 0; i < n; i++) {  // put zeros in the lower triangle
    for (int j = 0; j < i; j++) {
      ch[i][j] = 0.;
    }
  }

  return ch;
}







