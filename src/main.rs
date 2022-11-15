mod lib;

use plotpy::{Plot, Curve, Contour, Canvas, AsMatrix};
use russell_lab::generate3d;
use regex::Regex;

use std::collections::HashMap;

use crate::lib::spatial_vec::SpatialVec;
use crate::lib::cg::CG;
use crate::lib::optimizer::{Optimizer, DescentFrame};
use crate::lib::rosen::RosenDerivatives;
use crate::lib::tracer::Tracer;

fn gen_table(frames: Vec<DescentFrame>)
{ 
    for i in 0..frames.len() {
        let step = frames.get(i);
        match step {
            Some(frame) => println!("step {}: f{} = {:?}", i, frame.position, frame.value),
            None => println!("oops"),
        }
    }
}

struct FakeMatrix(Vec<SpatialVec<2>>);

impl<'a> AsMatrix<'a, f64> for FakeMatrix {
    fn at(&self, i: usize, j: usize) -> f64 {
        let FakeMatrix(v) = self;
        v[i][j]
    }
    fn size(&self) -> (usize, usize) {
        let FakeMatrix(v) = self;
        (v.len(), 2)
    }
}

fn generate_artifacts(trial_results: &mut HashMap<String, Vec<DescentFrame>>, tracer: &mut Tracer) {
    let re = Regex::new(r"^(?P<alg>(.*))/(?P<coeff>.*)$").unwrap();

    let n = 200;
    for (trial_name, frames) in trial_results {
        let cap = re.captures(trial_name).unwrap();
        let rosen_coefficient = cap["coeff"].parse::<f64>().unwrap();
        let alg = &cap["alg"];
        let num_evals = tracer.calls.get(&format!("{}/objective", trial_name)).unwrap();

        let trial_key = &format!("{}-{}", alg, rosen_coefficient);

        // TODO: break this out into a matlab specific formatter
        println!("{}: ${}$ & ${}$ & ${}$", trial_key, frames.last().unwrap().value, frames.len() - 1, num_evals);

        // value graphing
        let mut grad_curve = Curve::new();
        let mut objective_curve = Curve::new();
        grad_curve.set_line_width(2.0);
        objective_curve.set_line_width(2.0);

        // add points
        grad_curve.points_begin();
        objective_curve.points_begin();
        for i in 0..frames.len() {
            let frame = frames[i];
            grad_curve.points_add(i as f64, frame.grad.norm());
            objective_curve.points_add(i as f64, frame.value);
        }
        grad_curve.points_end();
        objective_curve.points_end();

        // add curve to plot
        let mut grad_plot = Plot::new();
        grad_plot.add(&grad_curve).grid_and_labels("Iteration", "Grad Norm");

        let mut objective_plot = Plot::new();
        objective_plot.add(&objective_curve).grid_and_labels("Iteration", "Objective Value");

        grad_plot.save(&format!("./figs/{}/{0}-Grad-plot", trial_key));
        objective_plot.save(&format!("./figs/{}/{0}-objective-plot", trial_key));

        // path graphing
        let func = move |x: SpatialVec<2>| rosen_coefficient * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);
        let (x, y, z) = generate3d(-2.0, 2.0, -2.0, 2.0, n, n, |x1, x2| func(SpatialVec([x1, x2])));
        let mut contour = Contour::new();
        contour
            .set_colorbar_label("cost")
            .set_colormap_name("terrain")
            .set_selected_line_color("#f1eb67")
            .set_selected_line_width(12.0)
            .set_selected_level(0.0, true);

        // draw contour
        contour.draw(&x, &y, &z);

        let mut canvas = Canvas::new();
        canvas.set_face_color("None").set_edge_color("red");

        let mut points = vec![];
        for pos in frames.iter().map(|frame| frame.position) {
            points.push(pos);
        }
        canvas.draw_polyline(&FakeMatrix(points), false);

        // add contour to plot
        let mut plot = Plot::new();
        plot.add(&contour)
            .add(&canvas)
            .set_labels("x1", "x2");

        // save figure
        plot.save(&format!("./figs/{}/{0}-path", trial_key));
    }

}
fn main()
{
    let tracer= &mut Tracer::new();
    let rosen_coefficients = [1.0, 100.0];

    let trial_results: &mut HashMap<String, Vec<DescentFrame>> = &mut HashMap::new();
    let start = SpatialVec([-1.2, 1.0]);
    //let start = SpatialVec([5.0, 5.0]);

    for rosen_coefficient in rosen_coefficients {
        let func = |x: SpatialVec<2>| {
            let rosen_coefficient = rosen_coefficient;
            rosen_coefficient * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2)
        };
        let derivs = RosenDerivatives::new(rosen_coefficient);

        let optimizer = &mut Optimizer {
            initial_step_length: 1.0,
            step_contraction_factor: 0.5,
            step_threshold_coefficient: 0.01,
            objective: &func,
            derivs: RosenDerivatives::clone(&derivs),
            tracer
        };

        let grad_predicate = move |data: DescentFrame| data.grad.norm() > 1.0e-3;

        let grad_trial_name = format!("Grad_Descent/{}", rosen_coefficient);
        trial_results.insert(
          grad_trial_name.clone(),
          Optimizer::descent(
                optimizer,
                grad_trial_name,
                start.clone(),
                &grad_predicate,
                &mut |x: SpatialVec<2>| -RosenDerivatives::gen_grad(&derivs, x),
            )
        );

        let newton_trial_name = format!("Newton_Iteration/{}", rosen_coefficient);
        trial_results.insert(
          newton_trial_name.clone(),
          Optimizer::descent(
                optimizer,
                newton_trial_name,
                start.clone(),
                &grad_predicate,
                &mut |x: SpatialVec<2>| -RosenDerivatives::gen_newton(&derivs, x),
            )
        );

        let cg_trial_name = format!("CG_iteration/{}", rosen_coefficient);
        let cg_tracer= &mut Tracer::new();
        // TODO: move the direction gen strategy into the optimizer
        let cg = &mut CG {
            current_direction: SpatialVec([0.0, 0.0]),
            current_location: start.clone(),
            derivs,
            trial_name: cg_trial_name.clone(),
            tracer: cg_tracer,
        };

        trial_results.insert(
            cg_trial_name.clone(),
            Optimizer::descent(
                optimizer,
                cg_trial_name,
                start.clone(),
                &grad_predicate,
                &mut |x: SpatialVec<2>| {
                    cg.gen_direction(x)
                }
            )
        );
        tracer.merge(cg.tracer);
    }

    generate_artifacts(trial_results, tracer);

}
