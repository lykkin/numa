use crate::lib::spatial_vec::SpatialVec;
// TODO: break this into its own file with some tests
// TODO: make this into a generic interface
#[derive(Clone)]
pub struct RosenDerivatives<'a, const SIZE: usize>
{
    pub rosen_coefficient: f64,
    pub grad: &'a dyn Fn(SpatialVec<SIZE>) -> SpatialVec<SIZE>,
    pub newton: &'a dyn Fn(SpatialVec<SIZE>) -> SpatialVec<SIZE>
}

// TODO: cache the values per struct
impl<const SIZE: usize> RosenDerivatives<'_, SIZE>
{
    // TODO: implement numerical grad method
    pub fn gen_grad(self: &Self, x: SpatialVec<SIZE>) -> SpatialVec<SIZE>
    {
        (self.grad)(x)
    }

    pub fn gen_newton(self: &Self, x: SpatialVec<SIZE>) -> SpatialVec<SIZE>
    {
        (self.newton)(x)
    }

}