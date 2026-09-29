//! Rust language runner: radiate (native) x every problem in the spec x every seed.
//!
//! Invoked by `rust/run.sh`; reads `--spec` and writes `runs.csv`, `history.csv` and
//! `fronts.csv` to `--out` in the layout described by `schema.md`. Operator choices mirror
//! `python/adapters/radiate_adapter.py`, so radiate (rust) vs radiate (python) differ only
//! in where the engine and the fitness function run.

mod problems;

use problems::{Kind, Problem};
use radiate::*;
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

const LANGUAGE: &str = "rust";
const LIBRARY: &str = "radiate";

struct Config {
    population_size: usize,
    generations: usize,
    crossover_rate: f32,
    mutation_rate: f32,
    seeds: Vec<u64>,
    problems: Vec<String>,
    spec_hash: String,
}

struct RunOutput {
    best_fitness: Option<f64>,
    best_solution: Vec<String>,
    history: Vec<f64>,
    wall_time_s: f64,
    front: Vec<Vec<f64>>,
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {path:?}: {e}"))
}

fn load_config(spec_dir: &Path) -> Config {
    let raw = read_json(&spec_dir.join("config.json"));
    let strings = |key: &str| -> Vec<String> {
        raw[key].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
    };
    Config {
        population_size: raw["population_size"].as_u64().unwrap() as usize,
        generations: raw["generations"].as_u64().unwrap() as usize,
        crossover_rate: raw["crossover_rate"].as_f64().unwrap() as f32,
        mutation_rate: raw["mutation_rate"].as_f64().unwrap() as f32,
        seeds: raw["seeds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect(),
        problems: strings("problems"),
        spec_hash: raw["spec_hash"].as_str().unwrap().to_string(),
    }
}

/// Runs a single-objective engine for `generations` epochs. The timer covers only the
/// evolution loop (including the first population evaluation), per schema.md.
fn run_single<C, T>(
    engine: GeneticEngine<C, T>,
    generations: usize,
    minimize: bool,
    solution: impl Fn(&T) -> Vec<String>,
) -> RunOutput
where
    C: Chromosome + Clone + 'static,
    T: Clone + Send + Sync + 'static,
{
    let mut history = Vec::with_capacity(generations);
    let mut best = if minimize { f64::INFINITY } else { f64::NEG_INFINITY };
    let mut last = None;

    let t0 = Instant::now();
    for generation in engine.iter().limit(generations) {
        let score = generation.score().as_f32() as f64;
        best = if minimize { best.min(score) } else { best.max(score) };
        history.push(best);
        last = Some(generation);
    }
    let wall_time_s = t0.elapsed().as_secs_f64();

    let last = last.expect("engine produced no generations");
    RunOutput {
        best_fitness: Some(last.score().as_f32() as f64),
        best_solution: solution(last.value()),
        history,
        wall_time_s,
        front: Vec::new(),
    }
}

fn run_mo<C, T>(engine: GeneticEngine<C, T>, generations: usize) -> RunOutput
where
    C: Chromosome + Clone + 'static,
    T: Clone + Send + Sync + 'static,
{
    let t0 = Instant::now();
    let last = engine.iter().limit(generations).last().expect("engine produced no generations");
    let wall_time_s = t0.elapsed().as_secs_f64();

    let front = last
        .front()
        .expect("multi-objective engine has no front")
        .values()
        .iter()
        .filter_map(|member| member.score())
        .map(|score| score.iter().map(|&v| v as f64).collect())
        .collect();

    RunOutput { best_fitness: None, best_solution: Vec::new(), history: Vec::new(), wall_time_s, front }
}

/// Genes are written at full precision (shortest round-trip form), so the aggregator
/// re-scores exactly the gene values the fitness function saw.
fn float_genes(genes: &Vec<f64>) -> Vec<String> {
    genes.iter().map(f64::to_string).collect()
}

/// `scoped_seed` reseeds this thread's RNG for the duration of the run, so each
/// (problem, seed) result is reproducible regardless of which runs came before it.
fn run(problem: &Problem, config: &Config, seed: u64) -> RunOutput {
    random_provider::scoped_seed(seed, || run_seeded(problem, config))
}

fn run_seeded(problem: &Problem, config: &Config) -> RunOutput {
    let pop = config.population_size;
    let gens = config.generations;
    let cr = config.crossover_rate;
    let mr = config.mutation_rate;

    match &problem.kind {
        Kind::Continuous { function, dim, bounds } => {
            let function = function.clone();
            let engine = GeneticEngine::builder()
                .codec(FloatCodec::vector(*dim, bounds.0..bounds.1))
                .minimizing()
                .population_size(pop)
                .offspring_selector(TournamentSelector::new(3))
                .survivor_selector(EliteSelector::new())
                .alter(alters!(UniformCrossover::new(cr), ArithmeticMutator::new(mr)))
                .fitness_fn(move |x: Vec<f64>| problems::continuous(&function, &x))
                .build();
            run_single(engine, gens, true, float_genes)
        }
        Kind::Knapsack { weights, values, capacity, penalty } => {
            let (weights, values, capacity, penalty) = (weights.clone(), values.clone(), *capacity, *penalty);
            let engine = GeneticEngine::builder()
                .codec(BitCodec::vector(weights.len()))
                .maximizing()
                .population_size(pop)
                .offspring_selector(TournamentSelector::new(3))
                .survivor_selector(EliteSelector::new())
                .alter(alters!(UniformCrossover::new(cr), UniformMutator::new(mr)))
                .fitness_fn(move |bits: Vec<bool>| problems::knapsack(&weights, &values, capacity, penalty, &bits))
                .build();
            run_single(engine, gens, false, |bits: &Vec<bool>| {
                bits.iter().map(|&b| (b as u8).to_string()).collect()
            })
        }
        Kind::Tsp { dist } => {
            let dist = dist.clone();
            let engine = GeneticEngine::builder()
                .codec(PermutationCodec::new((0..dist.len()).collect::<Vec<usize>>()))
                .minimizing()
                .population_size(pop)
                .offspring_selector(TournamentSelector::new(3))
                .survivor_selector(EliteSelector::new())
                .alter(alters!(PMXCrossover::new(cr), InversionMutator::new(mr)))
                .fitness_fn(move |tour: Vec<usize>| problems::tsp(&dist, &tour))
                .build();
            run_single(engine, gens, true, |tour: &Vec<usize>| tour.iter().map(usize::to_string).collect())
        }
        Kind::NQueens { n } => {
            let engine = GeneticEngine::builder()
                .codec(PermutationCodec::new((0..*n).collect::<Vec<usize>>()))
                .minimizing()
                .population_size(pop)
                .offspring_selector(TournamentSelector::new(3))
                .survivor_selector(EliteSelector::new())
                .alter(alters!(PMXCrossover::new(cr), InversionMutator::new(mr)))
                .fitness_fn(|perm: Vec<usize>| problems::nqueens(&perm))
                .build();
            run_single(engine, gens, true, |perm: &Vec<usize>| perm.iter().map(usize::to_string).collect())
        }
        Kind::Mo { function, n_var, n_obj, bounds } => {
            let (function, n_obj) = (function.clone(), *n_obj);
            // NSGA-II: crowded-comparison tournament for parents, rank + crowding for survivors,
            // SBX / polynomial mutation with eta=20, matching the Python adapters.
            let engine = GeneticEngine::builder()
                .codec(FloatCodec::vector(*n_var, bounds.0..bounds.1))
                .multi_objective(vec![Optimize::Minimize; n_obj])
                .front_size(pop..pop + 50)
                .population_size(pop)
                .offspring_selector(TournamentNSGA2Selector::new())
                .survivor_selector(NSGA2Selector::new())
                .offspring_fraction(0.5)
                .alter(alters!(
                    SimulatedBinaryCrossover::new(cr, 20.0),
                    PolynomialMutator::new(mr, 20.0)
                ))
                .fitness_fn(move |x: Vec<f64>| problems::multi_objective(&function, n_obj, &x))
                .build();
            run_mo(engine, gens)
        }
    }
}

fn write_csv(path: &Path, header: &str, rows: &[String]) {
    let mut text = String::with_capacity(rows.iter().map(|r| r.len() + 1).sum::<usize>() + header.len() + 1);
    text.push_str(header);
    text.push('\n');
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    std::fs::write(path, text).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
}

fn main() {
    let mut spec_dir = None;
    let mut out_dir = None;
    let mut library_version = String::from("unknown");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spec" => spec_dir = args.next().map(PathBuf::from),
            "--out" => out_dir = args.next().map(PathBuf::from),
            "--library-version" => library_version = args.next().expect("--library-version value"),
            other => panic!("unknown argument {other:?}"),
        }
    }
    let spec_dir = spec_dir.expect("--spec is required");
    let out_dir = out_dir.expect("--out is required");
    std::fs::create_dir_all(&out_dir).unwrap();

    let config = load_config(&spec_dir);
    let mut runs = Vec::new();
    let mut history = Vec::new();
    let mut fronts = Vec::new();

    for name in &config.problems {
        let spec = read_json(&spec_dir.join("problems").join(format!("{name}.json")));
        let problem = Problem::from_spec(&spec);
        for &seed in &config.seeds {
            print!("{:>12} | {:>8} | seed={seed} ", problem.name, LIBRARY);
            let out = run(&problem, &config, seed);

            let key = format!("{LANGUAGE},{LIBRARY},{},{seed}", problem.name);
            let timestamp = humantime::format_rfc3339_seconds(SystemTime::now());
            let best = out.best_fitness.map(|v| v.to_string()).unwrap_or_default();
            runs.push(format!(
                "{LANGUAGE},{LIBRARY},{library_version},{},{seed},{best},{},{},{},{timestamp}",
                problem.name,
                out.best_solution.join(";"),
                out.wall_time_s,
                config.spec_hash,
            ));
            for (generation, value) in out.history.iter().enumerate() {
                history.push(format!("{key},{generation},{value}"));
            }
            for (point, objectives) in out.front.iter().enumerate() {
                let mut joined = String::new();
                for (i, v) in objectives.iter().enumerate() {
                    let _ = write!(joined, "{}{v}", if i > 0 { ";" } else { "" });
                }
                fronts.push(format!("{key},{point},{joined}"));
            }

            match out.best_fitness {
                Some(v) => println!("-> best={v:.4}  time={:.3}s", out.wall_time_s),
                None => println!("-> front  time={:.3}s", out.wall_time_s),
            }
        }
    }

    write_csv(
        &out_dir.join("runs.csv"),
        "language,library,library_version,problem,seed,best_fitness,best_solution,wall_time_s,spec_hash,timestamp",
        &runs,
    );
    write_csv(&out_dir.join("history.csv"), "language,library,problem,seed,generation,best_so_far", &history);
    write_csv(&out_dir.join("fronts.csv"), "language,library,problem,seed,point,objectives", &fronts);
    println!("wrote {} runs to {}/", runs.len(), out_dir.display());
}
