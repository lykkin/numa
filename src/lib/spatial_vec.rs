use std::{ops::{Neg, SubAssign, AddAssign, Add, Mul, Sub, Index, IndexMut}, slice::SliceIndex};
use std::fmt;

// TODO: come up with a better name, it isn't really a spatial vector just had to avoid name collision
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SpatialVec<const SIZE: usize>(pub [f64; SIZE]);
impl<const SIZE: usize> SpatialVec<{SIZE}> {
    pub fn dot(self: Self, other: Self) -> f64 {
        let mut res = 0.0;
        for i in 0..SIZE {
            res += self[i] * other[i];
        }
        res
    }

    pub fn norm(self: &Self) -> f64 {
        let SpatialVec(data) = self;
        let mut res: f64 = 0.0;
        for d in data {
            res += d*d;
        }
        res.powf(0.5)
    }
}

impl<const SIZE: usize> Neg for SpatialVec<{SIZE}> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -1.0 * self
    }
}


impl<const SIZE: usize> fmt::Display for SpatialVec<{SIZE}> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(");
        for i in 0..SIZE-1 {
            write!(f, "{}, ", self[i]);
        }
        write!(f, "{})", self[SIZE - 1])
    }
}

impl<const SIZE: usize> SubAssign for SpatialVec<{SIZE}> {
    fn sub_assign(&mut self, other: Self) {
        for i in 0..SIZE {
            self[i] = self[i] - other[i];
        }
    }
}

impl<const SIZE: usize> AddAssign for SpatialVec<{SIZE}> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..SIZE {
            self[i] = self[i] + other[i];
        }
    }
}

impl<const SIZE: usize> From<[f64; SIZE]> for SpatialVec<{SIZE}>
{
    fn from(x: [f64; SIZE]) -> Self {
       Self(x) 
    }
}

impl<const SIZE: usize> Add for SpatialVec<{SIZE}>
{
    type Output = Self;
    fn add(self, other: Self) -> Self{
        let mut out : SpatialVec<SIZE> = SpatialVec([0.0;SIZE]);
        for i in 0..SIZE {
            out[i] = self[i] + other[i];
        }
        out
    }
}

impl<const SIZE: usize> Sub for SpatialVec<{SIZE}>
{
    type Output = Self;
    fn sub(self, other: Self) -> Self{
        let mut out : SpatialVec<SIZE> = SpatialVec([0.0;SIZE]);
        for i in 0..SIZE {
            out[i] = self[i] - other[i];
        }
        out
    }
}

impl<const SIZE: usize> Mul<f64> for SpatialVec<{SIZE}>
{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        let SpatialVec(data) = self;
        let mut out = [0.0;SIZE];
        for i in 0..SIZE {
            out[i] = data[i] * rhs;
        }
        SpatialVec(out)
    }
}

// multiplication isn't commutative OOTB so we have to define it manually
impl<const SIZE: usize> Mul<SpatialVec<{SIZE}>> for f64
{
    type Output = SpatialVec<SIZE>;
    fn mul(self, rhs: SpatialVec<SIZE>) -> Self::Output {
        rhs * self
    }
}

impl<const SIZE: usize, Idx> Index<Idx> for SpatialVec<{SIZE}>
where
    Idx: SliceIndex<[f64], Output = f64>,
{
    type Output = f64;
    fn index(&self, i: Idx) -> &Self::Output {
        let SpatialVec(data) = self;
        &data[i]
    }
}

impl<const SIZE: usize, Idx> IndexMut<Idx> for SpatialVec<{SIZE}> 
where
    Idx: SliceIndex<[f64], Output = f64>,
{
    fn index_mut(&mut self, i: Idx) -> &mut Self::Output {
        let SpatialVec(data) = self;
        &mut data[i]
    }
}