use crate::lib::spatial_vec::SpatialVec;
use crate::lib::rosen::RosenDerivatives;

use super::tracer::Tracer;
#[derive(Clone, Copy)]
pub struct DescentPredicateData
{
    pub steps: usize,
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
    fn calc_next_step_length(self: &mut Self, trial_name: &str, start:SpatialVec<2>, step_direction: SpatialVec<2>) -> f64 {
        let objective = self.objective;
        let start_value = objective(start);
        let grad = self.derivs.gen_grad(start);
        let candidate_dot = self.step_threshold_coefficient * step_direction.dot(grad);

        let mut step_size = self.initial_step_length;

        let mut num_loops = 0;
        while objective(start + step_size * step_direction) > start_value + step_size * candidate_dot {
            step_size *= self.step_contraction_factor;
            num_loops += 1;
        }
        self.tracer.increment_call(trial_name.to_owned() + "/grad", 1);
        self.tracer.increment_call(trial_name.to_owned() + "/objective", num_loops + 2);
        step_size
    }

    pub fn descent(self: &mut Self, trial_name: &str, start: SpatialVec<2>, should_step: &dyn Fn(DescentPredicateData) -> bool, direction_gen: &dyn Fn(SpatialVec<2>) -> SpatialVec<2>) -> Vec<SpatialVec<2>> {
        let objective = self.objective;

        let mut res: Vec<SpatialVec<2>> = vec![];
        let mut curr_step = start;
        let mut pred_data = DescentPredicateData {
            grad: self.derivs.gen_grad(curr_step),
            steps: 0,
            position: curr_step,
            value: objective(curr_step),
        };

        while should_step(pred_data) {
            res.push(SpatialVec::clone(&curr_step));

            let step_direction = direction_gen(curr_step);
            let step_length = Optimizer::calc_next_step_length(self, trial_name, curr_step, step_direction);

            curr_step += (step_length * step_direction);

            pred_data.grad = self.derivs.gen_grad(curr_step);
            pred_data.steps += 1;
            pred_data.position = curr_step;
            pred_data.value = objective(curr_step);
        }
        res.push(curr_step);
        self.tracer.increment_call(trial_name.to_owned() + "/grad", res.len());
        self.tracer.increment_call(trial_name.to_owned() + "/objective", res.len());
        res
    }
}
