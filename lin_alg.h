#include "tnt.h"
using namespace TNT;
using namespace std;

template <class T>  // create the nxn identity matrix
Array2D<T> identity(int n) {
  Array2D<T> I(n, n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i == j) {
        I[i][j] = 1;
      } else {
        I[i][j] = 0;
      }
  return I;
}

template <class T>  // scalar multiplication: aM
Array2D<T> scalarMult(T a, Array2D<T> M) {
  int m = M.dim1();
  int n = M.dim2();

  Array2D<T> N(m, n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) N[i][j] = a * M[i][j];

  return N;
}

template <class T>  // remove row a, column b from A (starting from 0)
Array2D<T> remove(Array2D<T> A, int a, int b) {
  int m = A.dim1();
  int n = A.dim2();

  Array2D<T> B(m - 1, n - 1);

  int u, v;

  for (int i = 0; i < m - 1; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (i < a) {
        u = i;
      } else {
        u = i + 1;
      }

      if (j < b) {
        v = j;
      } else {
        v = j + 1;
      }

      B[i][j] = A[u][v];
    }
  }

  return B;
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

template <class T>  // test if the error is less than ep(silon)
bool equal(Array2D<T> A, Array2D<T> B, T ep) {
  int m = A.dim1();
  int n = A.dim2();

  int r = B.dim1();
  int c = B.dim2();

  bool v = true;

  if (m == r && n == c) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        v = (B[i][j] - A[i][j] < ep);
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

template <class T>  // test if a matrix is symmetric
bool isSymm(Array2D<T> A, T err=0.000001) { 
  int m = A.dim1();
  int n = A.dim2();

  Array2D<T> B(n, m);

  B = transpose(A);

  return equal(A, B, err);
}

template <class T>  // 2x2 Cholesky
Array2D<T> twoChol(Array2D<T> A) {
  int m = A.dim1();
  int n = A.dim2();

  if (m != n) {
    throw runtime_error("The matrix is not square!");
  }

  Array2D<T> B(2, 2);

  B[0][0] = sqrt(A[0][0]);
  B[0][1] = A[0][1] / B[0][0];
  B[1][0] = 0.;
  B[1][1] = sqrt((A[0][0] * A[1][1] - A[0][1] * A[0][1]) / A[0][0]);
  return B;
}

template <class T>  // A->A_{1,2:n}
Array2D<T> firstUpperRow(Array2D<T> A) {
  int n = A.dim2();  // #columns
  Array2D<T> B(1, n - 1);

  for (int i = 0; i < n - 1; i++) {
    B[0][i] = A[0][i + 1];
  }
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

/////////// Cholesky 1 (just simple loops but the fastest)

template <class T>
Array2D<T> chol1(const Array2D<T>& A) {
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
      ch[j][i] = 0.;
    }

    for (int a = i + 1; a < n; a++) {
      for (int b = i + 1; b < n; b++) {
        ch[a][b] = ch[a][b] - ch[i][a] * ch[i][b];
      }
    }
  }

  return ch;
}

/////////////// Cholesky 2 - recursive

template <class T>
Array2D<T> chol2(const Array2D<T>& A) {
  const int m = A.dim1();
  const int n = A.dim2();

  if (m != n) {
    throw runtime_error("The matrix is not square!");
  }

  Array2D<T> ch(n, n);  // the Cholesky output

  if (n==1){ // the bottom case of a 1x1 matrix
      ch[0][0] = sqrt(A[0][0]);
      return ch;
  }

  Array2D<T> B(n - 1, n - 1);
  Array2D<T> R(1, n - 1);

  // B = A_{1:n-1,1:n-1}
  B = remove(A, 0, 0);  // the (0,0)-minor

  ch[0][0] = sqrt(A[0][0]);  // the top left entry
  // R = firstUpperRow(A); // first row, (0,0)-entry excluded
  // R = scalarMult(1/(ch[0][0]),firstUpperRow(A));

  for (int i = 0; i < n - 1; i++) {
    R[0][i] = A[0][i + 1] / ch[0][0];
  }

  for (int j = 1; j < n; j++) {  // put zeros below the top right entry
    ch[j][0] = 0.;
  }

  for (int j = 0; j < n - 1; j++) {
    ch[0][j + 1] = R[0][j];
  }

  // A_{1:n-1,1:n-1}-R_{0,1:n-1}^t*R_{0,1:n-1}
  Array2D<T> U(n - 1, n - 1);
  U = B - matmult(transpose(R), R); // reduce size

  U = chol2(U);  // recursion

  for (int i = 1; i < n; i++) {
    for (int j = 1; j < n; j++) {
      ch[i][j] = U[i - 1][j - 1];  
    }
  }

  return ch;
}


/////////// Cholesky 3 (simple loops - zeros copied at once)

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

//////////////

template <class T>
Array2D<T> bchol(const Array2D<T>& A) {
  const int m = A.dim1();
  const int n = A.dim2();
  const int d = 5;

  if (m != n) {
    throw runtime_error("The matrix is not square!");
  }

  Array2D<T> ch(n,n); // the output
  
  Array2D<T> U(d,d); // the top left dxd minor
  Array2D<T> tempM(n-d,n-d); // temp matrix  

  const int q = n / d; // the number of block operations
  const int r = n % d; // the remaining bit

  ch = A.copy(); // overwrite everything on the output

  T x = 0.; // temp variable

  int s; // the shift

  for (int p=0;p<q;p++){ // the big loop

    s = p*d; 

    for (int i=0; i<d; i++){ // extract the top-right minor 
      for (int j=0; j<d; j++){
        U[i][j] = ch[i+s][j+s]; // s gives the correct shift
      }
    }

    U = chol3(U); // compute a simple unblocked Cholesky: upper triangular U_{1,1}
  
    for (int i=0;i<d;i++){ // copy the top-left triangle back into the output
      for (int j=i;j<d;j++){
        ch[i+s][j+s] = U[i][j];      
      }
    }
  
    U = lower_tri_inverse(transpose(U)); // the inverse of the transpose of U_{1,1}
    
    for (int i=0;i<d;i++){  // compute the top-right triangle: U_{1,2} which is dx(n-(p+1)*d)
    for (int j=0;j<n-(p+1)*d;j++){
      x = 0.;
      for (int t=0;t<d;t++){
        x += U[i][t]*ch[t+s][j+s+d]; // Tricky shifts in (U_{1,2}^T)^(-1)*A_{1,2}
      }
      tempM[i][j] = x;
    }
  }  
  
  // copy entries into the local top-right part of the output and move on
  for (int i=0;i<d;i++){
  for (int j=0;j<n-(p+1)*d;j++){
    ch[i+s][j+s+d] = tempM[i][j];
  }
}
  
    for (int i=0;i<n-s-d;i++){ // updating the bottom-right triangle of size (n-(p+1)*d)
      for (int j=0;j<n-s-d;j++){
        x = 0.;    
        for (int t=0;t<d;t++){
          x += ch[t+s][i+s+d]*ch[t+s][j+s+d]; // an entry of U_{1,2}^T*U_{1,2}
        }
        tempM[i][j] = x; // tempM is U_{1,2}^T*U_{1,2}
      //ch[i+s+d][j+s+d] -= x; 
      }
    }

    for (int i=0;i<n-s-d;i++){ // updating the bottom-right triangle of size (n-(p+1)*d)
      for (int j=0;j<n-s-d;j++){
        ch[i+s+d][j+s+d] -= tempM[i][j]; // now ch is locally = A_{2,2}-U_{1,2}^T*U_{1,2}
      }}

  }

  if (r != 0){ // The last small block
    Array2D<T> V(n-d*q,n-d*q);

    for (int i=0;i<n-d*q;i++){
      for (int j=0;j<n-d*q;j++){
        V[i][j] = ch[i+d*q][j+d*q];
      }
    }
    V = chol3(V);

    for (int i=0;i<n-d*q;i++){
      for (int j=i;j<n-d*q;j++){
        ch[i+d*q][j+d*q] = V[i][j];
      }
    }
  }

  for (int i=0;i<n;i++){ // put zeros in the lower triangle
    for (int j=0;j<i;j++){
      ch[i][j] = 0.;
    }
  }

  return ch; 
}

//////////////Test

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








