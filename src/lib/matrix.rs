use std::{ops::{Neg, SubAssign, AddAssign, Add, Mul, Sub, Index, IndexMut}, slice::SliceIndex};
use std::fmt;

use crate::lib::spatial_vec::SpatialVec;
#[derive(Debug, Copy, Clone)]
// m[column][row]
pub struct Matrix<const SIZE: usize>(pub [SpatialVec<SIZE>; SIZE]);
impl<const SIZE: usize> Matrix<{SIZE}> {
  // binary exponentiation
  pub fn pow(self, mut n: i32) -> Self {
    let mut out = Matrix::identity();
    let mut base = self;
    while n > 0 {
      if n % 2 == 1 {
        out = out * base;
      }
      base = base * base;
      n = n / 2;
    }
    out
  }

  pub fn identity() -> Self {
    let mut out = [SpatialVec([0.0; SIZE]); SIZE];
    for i in 0..SIZE {
      out[i][i] = 1.0;
    }
    Matrix(out)
  }

  pub fn transpose(self) -> Self {
    let mut out = [SpatialVec([0.0; SIZE]); SIZE];
    let Matrix(d) = self;
    for i in 0..SIZE {
      for j in 0..SIZE {
        out[i][j] = d[j][i];
      }
    }
    Matrix(out)
  }

  pub fn is_symmetric(self) -> bool {
    let Matrix(d) = self;
    for i in 0..SIZE {
      for j in i+1..SIZE {
        if d[i][j] != d[j][i] {
          return false
        }
      }
    }
    true
  }
}

impl<const SIZE: usize> Mul<Self> for Matrix<SIZE>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
      let Matrix(data) = self;
      let Matrix(other) = rhs;
      let mut out = [SpatialVec([0.0; SIZE]); SIZE];
      for column in 0..SIZE {
        for row in 0..SIZE {
          for i in 0..SIZE {
            out[column][row] += data[column][i]*other[i][row];
          }
        }
      }
      Matrix(out)
    }
}

impl<const SIZE: usize> Mul<SpatialVec<SIZE>> for Matrix<SIZE>
{
    type Output = SpatialVec<SIZE>;
    fn mul(self, rhs: SpatialVec<SIZE>) -> Self::Output {
      let Matrix(data) = self;
      let mut out = SpatialVec([0.0;SIZE]);
      for i in 0..SIZE {
        out += rhs[i]*data[i];
      }
      out
    }
}


impl<const SIZE: usize> Mul<Matrix<SIZE>> for f64
{
    type Output = Matrix<SIZE>;
    fn mul(self, rhs: Matrix<SIZE>) -> Self::Output {
      let Matrix(data) = rhs;
      let mut out = [SpatialVec([0.0;SIZE]);SIZE];
      for column in 0..SIZE {
        for row in 0..SIZE {
          out[column][row] = self*data[column][row];
        }
      }
      Matrix(out)
    }
}

impl<const SIZE: usize> Add for Matrix<{SIZE}>
{
    type Output = Self;
    fn add(self, other: Self) -> Self{
        let mut out = [SpatialVec([0.0; SIZE]); SIZE];
        let Matrix(d) = self;
        let Matrix(o) = other;
        for i in 0..SIZE {
          for j in 0..SIZE {
            out[i][j] = d[i][j] + o[i][j];
          }
        }
        Matrix(out)
    }
}

impl<const SIZE: usize> Sub for Matrix<{SIZE}>
{
    type Output = Self;
    fn sub(self, other: Self) -> Self{
        let mut out = [SpatialVec([0.0; SIZE]); SIZE];
        let Matrix(d) = self;
        let Matrix(o) = other;
        for i in 0..SIZE {
          for j in 0..SIZE {
            out[i][j] = d[i][j] - o[i][j];
          }
        }
        Matrix(out)
    }
}

impl<const SIZE: usize> fmt::Display for Matrix<{SIZE}> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      let Matrix(data) = self;
      write!(f, "(");
      for j in 0..SIZE {
        let column = data[j];
        write!(f, "(");
        for i in 0..SIZE {
          write!(f, "{}", column[i]);
          if i != SIZE - 1 {
            write!(f, ", ");
          }
        }
        write!(f, ")");
        if j != SIZE - 1 {
          write!(f, ",");
        }
      }
      write!(f, ")")
    }
}

impl<const SIZE: usize> PartialEq<Self> for Matrix<SIZE> {
  fn eq(&self, other: &Self) -> bool {
    let Matrix(data) = self;
    let Matrix(o) = other;
    for i in 0..SIZE {
      for j in 0..SIZE {
        if data[i][j] != o[i][j] {
          return false
        }
      }
    }
    true
  }
}