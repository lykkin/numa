mod lib;
use crate::lib::spatial_vec::SpatialVec;
use crate::lib::optimizer::{Optimizer, DescentPredicateData};
use crate::lib::rosen::RosenDerivatives;

fn gen_latex_table(steps: Vec<SpatialVec<2>>, func: &dyn Fn(SpatialVec<2>) -> f64)
{ 
    for i in 0..steps.len() {
        let step = steps.get(i);
        match step {
            Some(step) => println!("{} & {:?} & {:?} & {:?} \\\\ \n \\hline", i, step[0], step[1], func(*step)),
            None => println!("oops"),
        }
    }
}

fn gen_table(steps: Vec<SpatialVec<2>>, func: &dyn Fn(SpatialVec<2>) -> f64)
{ 
    for i in 0..steps.len() {
        let step = steps.get(i);
        match step {
            Some(step) => println!("step {}: f({:?}, {:?}) = {:?}", i, step[0], step[1], func(*step)),
            None => println!("oops"),
        }
    }
}

fn main()
{
    let rosen_coefficient = 1.0;
    let func = |x: SpatialVec<2>| rosen_coefficient * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);
    let derivs = RosenDerivatives::new(rosen_coefficient);
    let optimizer = &Optimizer {
        initial_step_length: 1.0,
        step_contraction_factor: 0.5,
        step_threshold_coefficient: 0.01,
        objective: &func,
        derivs: RosenDerivatives::clone(&derivs)
    };

    let step_predicate = |data: DescentPredicateData| data.steps < 5;
    let grad_predicate = |data: DescentPredicateData| data.grad.norm() > 1.0e-3;

    println!("GRAD DESCENT");
    gen_table(
        Optimizer::descent(
            optimizer,
            SpatialVec([-1.2, 1.0]),
            //&step_predicate,
            &grad_predicate,
            &|x: SpatialVec<2>| -RosenDerivatives::gen_grad(&derivs, x),
        ),
        &func,
    );

    println!("\n\n");
    println!("NEWTON ITERATION");
    gen_table(
        Optimizer::descent(
            optimizer,
            SpatialVec([-1.2, 1.0]),
            //&step_predicate,
            &grad_predicate,
            &|x: SpatialVec<2>| -RosenDerivatives::gen_newton(&derivs, x),
        ),
        &func,
    );


}
