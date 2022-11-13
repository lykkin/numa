use super::spatial_vec::SpatialVec;
use super::rosen::RosenDerivatives;
use super::tracer::Tracer;

//TODO: generic dimensions
//TODO: scope down on access, add a constructor
pub struct CG<'a> {
    pub current_direction: SpatialVec<2>,
    pub current_location: SpatialVec<2>,
    pub derivs: RosenDerivatives,
    pub trial_name: String,
    pub tracer: &'a mut Tracer
}

impl CG<'_> {
    pub fn gen_direction(&mut self, x: SpatialVec<2>) -> SpatialVec<2> {
        let curr_grad = self.derivs.gen_grad(self.current_location);
        let next_grad = self.derivs.gen_grad(x);

        let beta = (next_grad.dot(next_grad))/(curr_grad.dot(curr_grad));

        self.current_direction = -next_grad + beta * self.current_direction;

        println!("{}", next_grad.dot(self.current_direction));
        if next_grad.dot(self.current_direction) >= 0.0 {
            self.current_direction = -next_grad;
            self.tracer.increment_call(
                self.trial_name.to_owned() + "/reset_direction",
                1
            );
        }
        self.tracer.increment_call(self.trial_name.to_owned() + "/grad", 2);
        self.current_direction
    }
}