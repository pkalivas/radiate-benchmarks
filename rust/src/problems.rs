//! Rust ports of the reference fitness functions in `radiate_benchmarks/problems/`.
//!
//! Float genes are `f64`, matching radiate's Python bindings (whose float codec is f64), so
//! radiate (rust) and radiate (python) run the same engine configuration.

use serde_json::Value;
use std::f64::consts::{E, PI};

pub struct Problem {
    pub name: String,
    pub kind: Kind,
}

pub enum Kind {
    Continuous { function: String, dim: usize, bounds: (f64, f64) },
    Knapsack { weights: Vec<f64>, values: Vec<f64>, capacity: f64, penalty: f64 },
    Tsp { dist: Vec<Vec<f64>> },
    NQueens { n: usize },
    Mo { function: String, n_var: usize, n_obj: usize, bounds: (f64, f64) },
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("expected a JSON array").iter().map(|x| x.as_f64().unwrap()).collect()
}

fn bounds(v: &Value) -> (f64, f64) {
    let b = floats(v);
    (b[0], b[1])
}

impl Problem {
    pub fn from_spec(spec: &Value) -> Self {
        let name = spec["name"].as_str().unwrap().to_string();
        let usize_of = |key: &str| spec[key].as_u64().unwrap() as usize;
        let kind = match spec["kind"].as_str().unwrap() {
            "continuous" => Kind::Continuous {
                function: spec["function"].as_str().unwrap().to_string(),
                dim: usize_of("dim"),
                bounds: bounds(&spec["bounds"]),
            },
            "knapsack" => Kind::Knapsack {
                weights: floats(&spec["weights"]),
                values: floats(&spec["values"]),
                capacity: spec["capacity"].as_f64().unwrap(),
                penalty: spec["overweight_penalty"].as_f64().unwrap(),
            },
            "tsp" => {
                let coords: Vec<Vec<f64>> =
                    spec["coords"].as_array().unwrap().iter().map(floats).collect();
                let dist = coords
                    .iter()
                    .map(|a| {
                        coords
                            .iter()
                            .map(|b| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
                            .collect()
                    })
                    .collect();
                Kind::Tsp { dist }
            }
            "nqueens" => Kind::NQueens { n: usize_of("n") },
            "mo" => Kind::Mo {
                function: spec["function"].as_str().unwrap().to_string(),
                n_var: usize_of("n_var"),
                n_obj: usize_of("n_obj"),
                bounds: bounds(&spec["bounds"]),
            },
            other => panic!("unknown problem kind {other:?}"),
        };
        Problem { name, kind }
    }
}

pub fn continuous(function: &str, x: &[f64]) -> f64 {
    let n = x.len() as f64;
    match function {
        "sphere" => x.iter().map(|v| v * v).sum(),
        "rastrigin" => 10.0 * n + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>(),
        "rosenbrock" => x
            .windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum(),
        "ackley" => {
            let sum1: f64 = x.iter().map(|v| v * v).sum();
            let sum2: f64 = x.iter().map(|v| (2.0 * PI * v).cos()).sum();
            -20.0 * (-0.2 * (sum1 / n).sqrt()).exp() - (sum2 / n).exp() + 20.0 + E
        }
        other => panic!("unknown continuous function {other:?}"),
    }
}

pub fn knapsack(weights: &[f64], values: &[f64], capacity: f64, penalty: f64, bits: &[bool]) -> f64 {
    let mut total_weight = 0.0;
    let mut total_value = 0.0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            total_weight += weights[i];
            total_value += values[i];
        }
    }
    if total_weight > capacity {
        total_value - penalty * (total_weight - capacity)
    } else {
        total_value
    }
}

/// Length of the closed tour (includes the leg from the last city back to the first).
pub fn tsp(dist: &[Vec<f64>], tour: &[usize]) -> f64 {
    (0..tour.len()).map(|i| dist[tour[i]][tour[(i + 1) % tour.len()]]).sum()
}

/// Number of diagonally attacking pairs; `perm[i]` is the row of the queen in column `i`.
pub fn nqueens(perm: &[usize]) -> f64 {
    let mut conflicts = 0;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i].abs_diff(perm[j]) == i.abs_diff(j) {
                conflicts += 1;
            }
        }
    }
    conflicts as f64
}

pub fn multi_objective(function: &str, n_obj: usize, x: &[f64]) -> Vec<f64> {
    match function {
        "zdt1" | "zdt3" => {
            let f1 = x[0];
            let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
            let h = if function == "zdt1" {
                1.0 - (f1 / g).sqrt()
            } else {
                1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin()
            };
            vec![f1, g * h]
        }
        "dtlz2" => {
            let m = n_obj;
            let g: f64 = x[m - 1..].iter().map(|v| (v - 0.5).powi(2)).sum();
            (0..m)
                .map(|i| {
                    let mut val = 1.0 + g;
                    for xj in &x[..m - 1 - i] {
                        val *= (xj * PI / 2.0).cos();
                    }
                    if i > 0 {
                        val *= (x[m - 1 - i] * PI / 2.0).sin();
                    }
                    val
                })
                .collect()
        }
        other => panic!("unknown multi-objective function {other:?}"),
    }
}
