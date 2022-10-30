use std::{ops::{Add, Mul, Sub, Index, IndexMut}, slice::SliceIndex};
#[derive(Debug, Copy, Clone, PartialEq)]
struct SpatialVec<const SIZE: usize>([f64; SIZE]);
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

// TODO: break this into its own file with some tests
struct Optimizer
{
    initial_step_length: f64,
    step_contraction_factor: f64,
    step_threshold_coefficient: f64
}

impl Optimizer 
{
    fn new(
        initial_step_length: f64,
        step_contraction_factor: f64,
        step_threshold_coefficient: f64
    ) -> Self {
        Self { initial_step_length, step_contraction_factor, step_threshold_coefficient }
    }

    fn calc_next_step_length(self: &Optimizer, start:SpatialVec<2>, step_direction: SpatialVec<2>) -> f64 {
        start + step_direction;
        0.0
    }

    fn descent(self: &Optimizer, start: SpatialVec<2>, step_length: f64, num_steps: usize, direction_gen: &dyn Fn(SpatialVec<2>) -> SpatialVec<2>) -> Vec<SpatialVec<2>> {
        let mut res: Vec<SpatialVec<2>> = vec![SpatialVec([0.0, 0.0]); num_steps + 1];
        let mut curr_step = start;
        for i in 0..=num_steps {
            res[i] = curr_step;
            let step_direction = direction_gen(curr_step);
            curr_step = curr_step - (step_length * step_direction);
        }
        res
    }
}

// TODO: break this into its own file with some tests
#[derive(Debug)]
struct Derivatives
{
    rosen_coefficient: f64
}

// TODO: cache the values per struct
impl Derivatives
{
    fn new(rosen_coefficient: f64) -> Self
    {
        Self{rosen_coefficient}
    }

    // TODO: implement numerical grad method
    fn gen_grad(self: &Derivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        SpatialVec([
            self.rosen_coefficient*-4.0*x[0] * (x[1] - x[0].powi(2)) - 2.0*(1.0  - x[0]),
            self.rosen_coefficient*2.0 * (x[1] - x[0].powi(2)),
        ])
    }

    fn gen_newton(self: &Derivatives, x: SpatialVec<2>) -> SpatialVec<2>
    {
        let grad = Derivatives::gen_grad(self, x);
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

fn main()
{
    let optimizer = &Optimizer::new(
        1.0,
        0.5,
        0.01
    );
    let derivs = &Derivatives::new(100.0);
    let func = |x: SpatialVec<2>| 100.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);

    println!("GRAD DESCENT");
    gen_latex_table(
        Optimizer::descent(
            optimizer,
            SpatialVec([0.0,0.0]),
            1.0,
            5,
            &|x: SpatialVec<2>| Derivatives::gen_grad(derivs, x),
        ),
        &func,
    );

    println!("\n\n");
    println!("NEWTON ITERATION");
    gen_latex_table(
        Optimizer::descent(
            optimizer,
            SpatialVec([0.0,0.0]),
            1.0,
            5,
            &|x: SpatialVec<2>| Derivatives::gen_newton(derivs, x),
        ),
        &func,
    );


}
