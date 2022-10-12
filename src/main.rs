type SpatialVec<const SIZE: usize> = [f64; SIZE];

fn descent(start: SpatialVec<2>, step_length: f64, num_steps: usize, direction_gen: &dyn Fn(SpatialVec<2>) -> SpatialVec<2>) -> Vec<SpatialVec<2>> {
    let mut res: Vec<SpatialVec<2>> = vec![[0.0, 0.0]; num_steps + 1];
    let mut curr_step = start;
    for i in 0..=num_steps {
        res[i] = curr_step;
        let step_direction = direction_gen(curr_step);
        curr_step[0] -= step_length * step_direction[0];
        curr_step[1] -= step_length * step_direction[1];
    }
    res
}

fn gen_grad(x: SpatialVec<2>) -> SpatialVec<2> {
    [
        -400.0*x[0] * (x[1] - x[0].powi(2)) - 2.0*(1.0  - x[0]),
        200.0 * (x[1] - x[0].powi(2)),
    ]
}

fn gen_newton(x: SpatialVec<2>) -> SpatialVec<2> {
    let grad = gen_grad(x);
    let a = -400.0 * x[1] + 1200.0 * x[0].powi(2) + 2.0;
    let b = -400.0 * x[0];
    let c = b;
    let d = 200.0;
    let scale = 1.0/(a*d - b*c);
    [
        scale * (d*grad[0] - b*grad[1]),
        scale * (a*grad[1] - c*grad[0]),
    ]
}

fn display_descent(steps: Vec<SpatialVec<2>>, func: &dyn Fn(SpatialVec<2>) -> f64) { 
    for i in 0..steps.len() {
        let step = steps.get(i);
        match step {
            Some(step) => println!("{} & {:?} & {:?} & {:?} \\\\ \n \\hline", i, step[0], step[1], func(*step)),
            None => println!("oops"),
        }
    }
}

fn main() {
    let func = |x: SpatialVec<2>| 100.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);
    println!("GRAD DESCENT");
    display_descent(
        descent(
            [0.0,0.0],
            1.0,
            5,
            &gen_grad
        ),
        &func,
    );

    //for step in steps {
    println!("\n\n");
    println!("NEWTON ITERATION");
    display_descent(
        descent(
            [0.0,0.0],
            1.0,
            5,
            &gen_newton
        ),
        &func,
    );


}
