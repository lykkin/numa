use crate::lib::spatial_vec::SpatialVec;
use crate::lib::rosen::RosenDerivatives;

use super::tracer::Tracer;
#[derive(Clone, Copy)]
pub struct DescentFrame
{
    pub step: usize,
    pub grad: SpatialVec<2>,
    pub position: SpatialVec<2>,
    pub value: f64
}

// TODO: break this into its own file with some tests
// TODO: make the dimension generic
pub struct Optimizer<'a>
{
    pub initial_step_length: f64,
    pub step_contraction_factor: f64,
    pub step_threshold_coefficient: f64,
    pub objective: &'a dyn Fn(SpatialVec<2>) -> f64,
    pub derivs: RosenDerivatives,
    pub tracer: &'a mut Tracer
}

impl Optimizer<'_>
{
    fn calc_next_step_length(self: &mut Self, trial_name: &str, start:SpatialVec<2>, step_direction: SpatialVec<2>, start_value: f64) -> (f64, f64) {
        let objective = self.objective;
        let grad = self.derivs.gen_grad(start);
        let candidate_dot = self.step_threshold_coefficient * step_direction.dot(grad);

        let mut step_size = self.initial_step_length;
        let mut next_value = objective(start + step_size * step_direction);
        while next_value > start_value + step_size * candidate_dot {
            self.tracer.increment_call(trial_name.to_owned() + "/objective", 1);
            step_size *= self.step_contraction_factor;
            next_value = objective(start + step_size * step_direction);
        }
        self.tracer.increment_call(trial_name.to_owned() + "/grad", 1);
        self.tracer.increment_call(trial_name.to_owned() + "/objective", 1);
        (step_size, next_value)
    }

    pub fn descent(self: &mut Self, trial_name: String, start: SpatialVec<2>, should_step: &dyn Fn(DescentFrame) -> bool, direction_gen: &mut dyn FnMut(SpatialVec<2>) -> SpatialVec<2>) -> Vec<DescentFrame> {
        let objective = self.objective;

        let mut curr_step = start;
        let mut res: Vec<DescentFrame> = vec![
            DescentFrame {
                grad: self.derivs.gen_grad(curr_step),
                step: 0,
                position: curr_step,
                value: objective(curr_step),
            }
        ];

        let mut last_frame = res[0];
        while should_step(last_frame) {
            let step_direction = direction_gen(curr_step);
            let (step_length, next_value) = Optimizer::calc_next_step_length(self, trial_name.as_ref(), curr_step, step_direction, last_frame.value);

            curr_step += step_length * step_direction;
            last_frame = DescentFrame {
                grad: self.derivs.gen_grad(curr_step),
                step: res.len(),
                position: curr_step,
                value: next_value,
            };

            res.push(last_frame);
        }
        self.tracer.increment_call(trial_name.to_owned() + "/grad", res.len());
        res
    }
}
