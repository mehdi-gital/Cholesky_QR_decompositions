// QR decomposition using Householder reflections
#include "tnt.h"
#include<tuple>
using namespace TNT;
using namespace std;

template <class T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

int del(int a, int b) { // Kronecker's delta
  int d;
  if (a == b) {
    d = 1;
  } else {
    d = 0;
  }
  return d;
}

template <class T>  // create the nxn identity matrix
Array2D<T> idMat(int n) {
  Array2D<T> I(n, n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) I[i][j] = del(i, j);
    
  return I;
}

template <class T>
T norm(Array1D<T> v, int k) {  // the norm of the first k coordinates of v
  T L;                         // the output
  for (int i = 0; i < k; i++) {
    L += v[i] * v[i];
  }
  L = sqrt(L);
  return L;
}

template <class T>
Array1D<T> hh(Array1D<T> y, int k) {  // the Householder vector of y[0:k]
  Array1D<T> v(k);
  v = y.copy();

  T L;  // the norm
  L = norm(v, k);

  v[0] += sgn(v[0]) * L;  // w = w + sgn(y[0])||y||e_1
  L = norm(v, k);         // ||w||, k = m-i

  for (int p = 0; p < k; p++) {  // v = normalize w
    v[p] = v[p] / L;
  }

  return v;
}

template <class T>  // The Householder matrix
Array2D<T> HH(Array1D<T> v, int i, int m) {
  Array2D<T> H(m, m);

  for (int a = 0; a < m; a++) {
    for (int b = 0; b < m; b++) {
      if (a >= i and b >= i) {
        H[a][b] = del(a, b) - 2 * v[a - i] * v[b - i];
      } else {
        H[a][b] = del(a, b);
      }
    }
  }
  return H;
}

template <class T>  // The R in the QR factorization
std::tuple<Array2D<T>, Array2D<T> > qr(Array2D<T> A) {
  const int m = A.dim1();
  const int n = A.dim2();
  Array2D<T> R(m, n);  // The R = Hn...H2H1A
  Array2D<T> H(m, m);  // Housholder reflections
  Array2D<T> Q(m, m);  // The Q = H1H2...Hn
  Array1D<T> y(m);
  T L;  // length of y
  R = A.copy();
  Q = idMat<double>(m);  // initialize Q

  for (int i = 0; i < n; i++) {  // THE BIG LOOP

    for (int p = 0; p < m - i;
         p++) {            // extracts the column including and below the ith
      y[p] = R[p + i][i];  // diagonal entry in A
    }

    y = hh(y, m - i);  // the Householder vector of y[0:m-i]
    H = HH(y, i, m);   // the Householder matrix
    R = matmult(H, R);
    Q = matmult(Q, H);
  }

  for (int a = 0; a < n;
       a++) {  // turn the signs whenever a diagonal entry of R is negative
    if (R[a][a] < 0) {
      for (int b = 0; b < n; b++) {
        R[a][b] = -R[a][b];  // row a of R
        Q[b][a] = -Q[b][a];  // column a of Q
      }
      for (int b = n; b < m; b++) {
        Q[b][a] = -Q[b][a];  // the rest of column a of Q
      }
    }
  }

  return make_tuple(Q, R);
}

