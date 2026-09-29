// C# language runner: GeneticSharp x every problem in the spec x every seed.
//
// Invoked by csharp/run.sh; reads --spec and writes runs.csv, history.csv and fronts.csv to
// --out as described in schema.md. GeneticSharp has no multi-objective support, so it skips
// the MO suite. Before the timed seeds of each problem, one untimed warm-up run lets the JIT
// tier up the hot paths so it doesn't land in the first seed's wall_time_s.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Bench;
using GeneticSharp;

const string Language = "csharp";
const string Library = "geneticsharp";
const int WarmupSeed = 0;

string? specDir = null, outDir = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--spec": specDir = args[++i]; break;
        case "--out": outDir = args[++i]; break;
        default: throw new ArgumentException($"unknown argument {args[i]}");
    }
}
if (specDir is null || outDir is null) throw new ArgumentException("--spec and --out are required");
Directory.CreateDirectory(outDir);

var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(specDir, "config.json"))).RootElement;
int populationSize = config.GetProperty("population_size").GetInt32();
int generations = config.GetProperty("generations").GetInt32();
float crossoverRate = (float)config.GetProperty("crossover_rate").GetDouble();
float mutationRate = (float)config.GetProperty("mutation_rate").GetDouble();
var seeds = config.GetProperty("seeds").EnumerateArray().Select(s => s.GetInt32()).ToList();
string specHash = config.GetProperty("spec_hash").GetString()!;
var version = typeof(GeneticAlgorithm).Assembly.GetName().Version!;
string libraryVersion = $"{version.Major}.{version.Minor}.{version.Build}";

var runs = new List<string>();
var history = new List<string>();
string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);

foreach (var name in config.GetProperty("problems").EnumerateArray().Select(p => p.GetString()!))
{
    var problem = Problem.FromSpec(
        JsonDocument.Parse(File.ReadAllText(Path.Combine(specDir, "problems", name + ".json"))).RootElement);
    if (problem.Kind == Kind.Mo)
    {
        Console.WriteLine($"{problem.Name,12} | {Library,12} | not supported, skipping");
        continue;
    }

    Run(problem, WarmupSeed);
    foreach (int seed in seeds)
    {
        Console.Write($"{problem.Name,12} | {Library,12} | seed={seed} ");
        var (best, solution, hist, wall) = Run(problem, seed);

        string key = $"{Language},{Library},{problem.Name},{seed}";
        runs.Add(string.Join(",", Language, Library, libraryVersion, problem.Name, seed, F(best),
            string.Join(";", solution), F(wall), specHash, DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")));
        for (int g = 0; g < hist.Count; g++) history.Add($"{key},{g},{F(hist[g])}");
        Console.WriteLine($"-> best={best:F4}  time={wall:F3}s");
    }
}

Write("runs.csv", "language,library,library_version,problem,seed,best_fitness,best_solution,wall_time_s,spec_hash,timestamp", runs);
Write("history.csv", "language,library,problem,seed,generation,best_so_far", history);
Write("fronts.csv", "language,library,problem,seed,point,objectives", []);
Console.WriteLine($"wrote {runs.Count} runs to {outDir}/");

void Write(string file, string header, List<string> rows)
{
    var text = new StringBuilder(header).Append('\n');
    foreach (var row in rows) text.Append(row).Append('\n');
    File.WriteAllText(Path.Combine(outDir!, file), text.ToString());
}

(double Best, List<string> Solution, List<double> History, double Wall) Run(Problem p, int seed)
{
    RandomizationProvider.Current = new FastRandomRandomization();
    FastRandomRandomization.ResetSeed(seed);

    // GeneticSharp always maximizes, so minimization problems are scored as -f and the
    // reported values are flipped back.
    double sign = p.Minimize ? -1.0 : 1.0;
    IChromosome adam;
    ICrossover crossover;
    IMutation mutation;
    Func<IChromosome, double> objective;
    Func<IChromosome, List<string>> decode;

    switch (p.Kind)
    {
        case Kind.Continuous:
        {
            // FloatingPointChromosome encodes each value as fixed-point bits, and a negative value
            // needs all 64 bits, where most bit flips just slam the gene to a bound. So genes live
            // on the shifted range [0, hi - lo] with the fewest bits for 4 decimal places, and lo
            // is added back when decoding. (GeneticSharp's own samples use non-negative ranges.)
            const int fractionDigits = 4;
            double span = p.Hi - p.Lo;
            int bits = (int)Math.Ceiling(Math.Log2(span * Math.Pow(10, fractionDigits) + 1));
            adam = new FloatingPointChromosome(
                Enumerable.Repeat(0.0, p.Size).ToArray(),
                Enumerable.Repeat(span, p.Size).ToArray(),
                Enumerable.Repeat(bits, p.Size).ToArray(),
                Enumerable.Repeat(fractionDigits, p.Size).ToArray());
            double[] X(IChromosome c) => ((FloatingPointChromosome)c).ToFloatingPoints().Select(v => v + p.Lo).ToArray();
            crossover = new UniformCrossover(0.5f);
            mutation = new FlipBitMutation();
            objective = c => Fitness.Continuous(p.Function!, X(c));
            decode = c => X(c).Select(F).ToList();
            break;
        }
        case Kind.Knapsack:
        {
            adam = new BitChromosome(p.Size);
            int[] Bits(IChromosome c) => c.GetGenes().Select(g => (int)g.Value).ToArray();
            crossover = new UniformCrossover(0.5f);
            mutation = new FlipBitMutation();
            objective = c => Fitness.Knapsack(p, Bits(c));
            decode = c => Bits(c).Select(b => b.ToString()).ToList();
            break;
        }
        default:
        {
            // OrderedCrossover + ReverseSequenceMutation, as in GeneticSharp's own TSP sample.
            adam = new PermutationChromosome(p.Size);
            int[] Perm(IChromosome c) => c.GetGenes().Select(g => (int)g.Value).ToArray();
            crossover = new OrderedCrossover();
            mutation = new ReverseSequenceMutation();
            objective = p.Kind == Kind.Tsp ? c => Fitness.Tsp(p, Perm(c)) : c => Fitness.NQueens(Perm(c));
            decode = c => Perm(c).Select(v => v.ToString()).ToList();
            break;
        }
    }

    var ga = new GeneticAlgorithm(
        new Population(populationSize, populationSize, adam),
        new FuncFitness(c => sign * objective(c)),
        new TournamentSelection(3),
        crossover,
        mutation)
    {
        Termination = new GenerationNumberTermination(generations),
        CrossoverProbability = crossoverRate,
        MutationProbability = mutationRate,
    };

    // GeneticSharp's default reinsertion keeps no elite, so the current generation's best can
    // be worse than an earlier one: track the best ever seen, and the genes that achieved it.
    var hist = new List<double>(generations);
    double bestInternal = double.NegativeInfinity;
    IChromosome? bestChromosome = null;
    ga.GenerationRan += (_, _) =>
    {
        var current = ga.BestChromosome;
        if (current.Fitness!.Value > bestInternal)
        {
            bestInternal = current.Fitness.Value;
            bestChromosome = current.Clone();
        }
        hist.Add(sign * bestInternal);
    };

    var sw = Stopwatch.StartNew();
    ga.Start();
    sw.Stop();

    return (sign * bestInternal, decode(bestChromosome!), hist, sw.Elapsed.TotalSeconds);
}

/// <summary>Fixed-length bit string for knapsack.</summary>
sealed class BitChromosome : BinaryChromosomeBase
{
    private readonly int _length;

    public BitChromosome(int length) : base(length)
    {
        _length = length;
        CreateGenes();
    }

    public override IChromosome CreateNew() => new BitChromosome(_length);
}

/// <summary>A random permutation of 0..n-1, built like GeneticSharp's TspChromosome.</summary>
sealed class PermutationChromosome : ChromosomeBase
{
    private readonly int _n;

    public PermutationChromosome(int n) : base(n)
    {
        _n = n;
        var values = RandomizationProvider.Current.GetUniqueInts(n, 0, n);
        for (int i = 0; i < n; i++) ReplaceGene(i, new Gene(values[i]));
    }

    public override Gene GenerateGene(int geneIndex) => new(RandomizationProvider.Current.GetInt(0, _n));

    public override IChromosome CreateNew() => new PermutationChromosome(_n);
}
