use std::{ops::{SubAssign, AddAssign, Add, Mul, Sub, Index, IndexMut}, slice::SliceIndex};
use std::fmt;


// TODO: come up with a better name, it isn't really a spatial vector just had to avoid name collision
#[derive(Debug, Copy, Clone, PartialEq)]
struct SpatialVec<const SIZE: usize>([f64; SIZE]);
impl<const SIZE: usize> SpatialVec<{SIZE}> {
    fn dot(self: Self, other: Self) -> f64 {
        let mut res = 0.0;
        for i in 0..SIZE {
            res += self[i] * other[i];
        }
        res
    }

    fn norm(self: &Self) -> f64 {
        let SpatialVec(data) = self;
        let mut res: f64 = 0.0;
        for d in data {
            res += d*d;
        }
        res.powf(0.5)
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
#[derive(Clone, Copy)]
struct DescentPredicateData
{
    steps: usize,
    grad: SpatialVec<2>,
    position: SpatialVec<2>,
    value: f64
}

// TODO: break this into its own file with some tests
// TODO: make the dimension generic
struct Optimizer<'a>
{
    initial_step_length: f64,
    step_contraction_factor: f64,
    step_threshold_coefficient: f64,
    objective: &'a dyn Fn(SpatialVec<2>) -> f64,
    derivs: RosenDerivatives
}

impl Optimizer<'_>
{
    fn calc_next_step_length(self: &Self, start:SpatialVec<2>, step_direction: SpatialVec<2>) -> f64 {
        let objective = self.objective;
        let start_value = objective(start);
        let grad = self.derivs.gen_grad(start);
        let candidate_dot = self.step_threshold_coefficient * step_direction.dot(grad);

        let mut step_size = self.initial_step_length;

        while objective(start + step_size * step_direction) > start_value + step_size * candidate_dot {
            step_size *= self.step_contraction_factor;
        }

        step_size
    }

    fn descent(self: &Self, start: SpatialVec<2>, should_step: &dyn Fn(DescentPredicateData) -> bool, direction_gen: &dyn Fn(SpatialVec<2>) -> SpatialVec<2>) -> Vec<SpatialVec<2>> {
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
            let step_length = Optimizer::calc_next_step_length(self, curr_step, step_direction);

            curr_step += (step_length * step_direction);

            pred_data.grad = self.derivs.gen_grad(curr_step);
            pred_data.steps += 1;
            pred_data.position = curr_step;
            pred_data.value = objective(curr_step);
        }
        res.push(curr_step);
        res
    }
}

// TODO: break this into its own file with some tests
// TODO: make this into a generic interface
#[derive(Debug, Clone)]
struct RosenDerivatives
{
    rosen_coefficient: f64
}

// TODO: cache the values per struct
impl RosenDerivatives
{
    fn new(rosen_coefficient: f64) -> Self
    {
        Self{rosen_coefficient}
    }

    // TODO: implement numerical grad method
    fn gen_grad(self: &RosenDerivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        SpatialVec([
            self.rosen_coefficient*-4.0*x[0] * (x[1] - x[0].powi(2)) - 2.0*(1.0  - x[0]),
            self.rosen_coefficient*2.0 * (x[1] - x[0].powi(2)),
        ])
    }

    fn gen_newton(self: &RosenDerivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        let grad = RosenDerivatives::gen_grad(self, x);

        let a = self.rosen_coefficient*-4.0 * x[1] + 1200.0 * x[0].powi(2) + 2.0;
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
            &|x: SpatialVec<2>| -1.0*RosenDerivatives::gen_grad(&derivs, x),
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
            &|x: SpatialVec<2>| -1.0*RosenDerivatives::gen_newton(&derivs, x),
        ),
        &func,
    );


}
