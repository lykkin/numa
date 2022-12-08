use super::matrix::Matrix;
use super::spatial_vec::SpatialVec;
#[derive(Clone, Copy)]
pub struct DescentFrame<const SIZE: usize>
{
    pub residual: SpatialVec<SIZE>,
    pub position: SpatialVec<SIZE>,
}

//TODO: generic dimensions
//TODO: scope down on access, add a constructor
//TODO: implement a shared interface
pub struct CG<const SIZE: usize> {
    pub direction: SpatialVec<SIZE>,
    pub location: SpatialVec<SIZE>,
    pub residual: SpatialVec<SIZE>,
}

impl<const SIZE: usize> CG<SIZE> {
    pub fn descent(self: &mut Self, should_step: &dyn Fn(DescentFrame<SIZE>) -> bool, A: Matrix<SIZE>) -> Vec<DescentFrame<SIZE>> {
        let mut res: Vec<DescentFrame<SIZE>> = vec![
            DescentFrame {
                residual: self.residual,
                position: self.location,
            }
        ];

        let mut last_frame = res[0];
        while should_step(last_frame) {
            let direction_image = A*self.direction;
            let residual_square = self.residual.dot_self();

            let step_length = residual_square/self.direction.dot(direction_image);

            self.location += step_length * self.direction;
            self.residual += step_length * direction_image;

            let beta = self.residual.dot_self()/residual_square;
            self.direction = -self.residual + beta * self.direction;
            last_frame = DescentFrame {
                residual: self.residual,
                position: self.location,
            };

            res.push(last_frame);
        }
        res
    }
}