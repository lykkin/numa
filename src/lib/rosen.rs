use crate::lib::spatial_vec::SpatialVec;
// TODO: break this into its own file with some tests
// TODO: make this into a generic interface
#[derive(Debug, Clone)]
pub struct RosenDerivatives
{
    rosen_coefficient: f64
}

// TODO: cache the values per struct
impl RosenDerivatives
{
    pub fn new(rosen_coefficient: f64) -> Self
    {
        Self{rosen_coefficient}
    }

    // TODO: implement numerical grad method
    pub fn gen_grad(self: &RosenDerivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        SpatialVec([
            self.rosen_coefficient*-4.0*x[0] * (x[1] - x[0].powi(2)) - 2.0*(1.0  - x[0]),
            self.rosen_coefficient*2.0 * (x[1] - x[0].powi(2)),
        ])
    }

    pub fn gen_newton(self: &RosenDerivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        let grad = RosenDerivatives::gen_grad(self, x);

        let a = self.rosen_coefficient*-4.0 * x[1] + 12.0 * self.rosen_coefficient * x[0].powi(2) + 2.0;
        let b = self.rosen_coefficient*-4.0 * x[0];
        let c = b;
        let d = self.rosen_coefficient*2.0;

        let scale = 1.0/(a*d - b*c);

        SpatialVec([
            scale * (d*grad[0] - b*grad[1]),
            scale * (a*grad[1] - c*grad[0]),
        ])
    }

}