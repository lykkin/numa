mod lib;

use image::{GrayImage, Luma};

use std::fs::File;
use std::io::{BufReader, BufRead};

use crate::lib::spatial_vec::SpatialVec;
use crate::lib::cg::{CG, DescentFrame};
use crate::lib::matrix::Matrix;

fn gen_matrix<const D: usize>(L: f64) -> Matrix<D>{
    let mut res = [SpatialVec([0.0; D]); D];
    for i in 0..D {
        res[i][i] = 1.0 - 2.0*L;
        if i < D-1 {
            res[i + 1][i] = L;
            res[i][i + 1] = L;
        }
    }
    Matrix(res)
}

fn tikhonov<const SIZE: usize>(A: Matrix<SIZE>, D: &Vec<SpatialVec<SIZE>>, lambda: f64) -> Vec<SpatialVec<SIZE>> {
    let At = A.transpose();
    let AAt = A*At;
    let reg_mat = AAt + lambda*lambda*Matrix::identity();

    let mut out: Vec<SpatialVec<SIZE>> = Vec::with_capacity(D.len());
    let mut start: SpatialVec<SIZE> = D[0];
    for i in 0..D.len() {
        let d = D[i];
        let Atd = At*d;

        let residual_predicate = |data: DescentFrame<SIZE>| data.residual.norm() > 1.0e-6;

        // TODO: move the direction gen strategy into the optimizer
        let resid = reg_mat*start - Atd;
        let cg = &mut CG {
            direction: -resid,
            location: start,
            residual: resid,
        };

        let res = cg.descent(
            &residual_predicate,
            reg_mat,
        );

        let unblurred = res.last().unwrap().position;
        start = unblurred;
        out.push(unblurred);
        println!("step:{} steps:{}", out.len(), res.len());
    }
    out
}

fn main()
{
    // load image data and arrange into vectors
    let file = File::open("./data/dollarblur.txt").unwrap();
    let reader = BufReader::new(file);
    let mut input_buffer = [[0.0; 220]; 520];

    let mut lineNum = 0;
    for line in reader.lines() {
        let mut cursor = 0;
        for val in line.unwrap().split("    ") {
            if val != "" {
                input_buffer[cursor][lineNum] = val.trim().parse::<f64>().unwrap();
                cursor += 1;
            }
        }
        lineNum += 1;
    }

    let mut D: Vec<SpatialVec<220>> = Vec::with_capacity(520);
    for i in 0..520 {
        D.push(SpatialVec(input_buffer[i]));
    }

    println!("printing blurred");
    let mut image = GrayImage::from_raw(520, 220, vec![0u8; 220*520]).unwrap();
    for y in 0..D.len() {
        let row = D[y];
        for x in 0..row.len() {
            image.put_pixel(y as u32, x as u32, Luma([row[x].round() as u8; 1]));
        }
    }

    let res = image.save(format!("./unblur/original.png"));
    match res {
        Err(e) => println!("{:?}", e),
        Ok(()) => println!("ok!")
    }

    // generate blur matrix
    let B = gen_matrix::<220>(0.45);
    let A = B.pow(25);

    let mut lambda = 1.0;
    while lambda > 0.000001 {
        let out = tikhonov(A, &D, lambda);

        println!("printing {}", lambda);
        let mut image = GrayImage::from_raw(520, 220, vec![0u8; 220*520]).unwrap();
        for y in 0..out.len() {
            let row = out[y];
            for x in 0..row.len() {
                image.put_pixel(y as u32, x as u32, Luma([row[x].round() as u8; 1]));
            }
        }

        let res = image.save(format!("./unblur/lambda_{}.png", lambda));
        match res {
            Err(e) => println!("{:?}", e),
            Ok(()) => println!("ok!")
        }
        lambda /= 10.0;
    }

}
