using System.Text.Json;

namespace Bench;

/// <summary>
/// C# ports of the reference fitness functions in <c>radiate_benchmarks/problems/</c>.
/// The aggregator re-scores every best solution with the Python reference, so these must
/// match it formula for formula.
/// </summary>
enum Kind { Continuous, Knapsack, Tsp, NQueens, Mo }

sealed record Problem(
    string Name,
    Kind Kind,
    string? Function,
    int Size,
    double Lo,
    double Hi,
    double[]? Weights,
    double[]? Values,
    double Capacity,
    double Penalty,
    double[,]? Dist,
    bool Minimize)
{
    public static Problem FromSpec(JsonElement spec)
    {
        string name = spec.GetProperty("name").GetString()!;
        bool minimize = spec.GetProperty("minimize").GetBoolean();
        double[] Doubles(string key) => spec.GetProperty(key).EnumerateArray().Select(v => v.GetDouble()).ToArray();

        switch (spec.GetProperty("kind").GetString())
        {
            case "continuous":
            {
                var b = Doubles("bounds");
                return new Problem(name, Kind.Continuous, spec.GetProperty("function").GetString(),
                    spec.GetProperty("dim").GetInt32(), b[0], b[1], null, null, 0, 0, null, minimize);
            }
            case "knapsack":
                return new Problem(name, Kind.Knapsack, null, spec.GetProperty("n_items").GetInt32(), 0, 0,
                    Doubles("weights"), Doubles("values"), spec.GetProperty("capacity").GetDouble(),
                    spec.GetProperty("overweight_penalty").GetDouble(), null, minimize);
            case "tsp":
            {
                var coords = spec.GetProperty("coords").EnumerateArray()
                    .Select(c => c.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray();
                int n = coords.Length;
                var dist = new double[n, n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double dx = coords[i][0] - coords[j][0], dy = coords[i][1] - coords[j][1];
                        dist[i, j] = Math.Sqrt(dx * dx + dy * dy);
                    }
                return new Problem(name, Kind.Tsp, null, n, 0, 0, null, null, 0, 0, dist, minimize);
            }
            case "nqueens":
                return new Problem(name, Kind.NQueens, null, spec.GetProperty("n").GetInt32(), 0, 0,
                    null, null, 0, 0, null, minimize);
            case "mo":
                return new Problem(name, Kind.Mo, spec.GetProperty("function").GetString(),
                    spec.GetProperty("n_var").GetInt32(), 0, 1, null, null, 0, 0, null, minimize);
            default:
                throw new ArgumentException($"unknown problem kind {spec.GetProperty("kind")}");
        }
    }
}

static class Fitness
{
    public static double Continuous(string function, double[] x)
    {
        int n = x.Length;
        switch (function)
        {
            case "sphere":
                return x.Sum(v => v * v);
            case "rastrigin":
                return 10.0 * n + x.Sum(v => v * v - 10.0 * Math.Cos(2 * Math.PI * v));
            case "rosenbrock":
            {
                double s = 0;
                for (int i = 0; i < n - 1; i++)
                    s += 100.0 * Math.Pow(x[i + 1] - x[i] * x[i], 2) + Math.Pow(1 - x[i], 2);
                return s;
            }
            case "ackley":
            {
                double sum1 = x.Sum(v => v * v);
                double sum2 = x.Sum(v => Math.Cos(2 * Math.PI * v));
                return -20 * Math.Exp(-0.2 * Math.Sqrt(sum1 / n)) - Math.Exp(sum2 / n) + 20 + Math.E;
            }
            default:
                throw new ArgumentException($"unknown continuous function {function}");
        }
    }

    public static double Knapsack(Problem p, int[] bits)
    {
        double weight = 0, value = 0;
        for (int i = 0; i < bits.Length; i++)
        {
            if (bits[i] == 1)
            {
                weight += p.Weights![i];
                value += p.Values![i];
            }
        }
        return weight > p.Capacity ? value - p.Penalty * (weight - p.Capacity) : value;
    }

    /// <summary>Length of the closed tour (includes the leg from the last city back to the first).</summary>
    public static double Tsp(Problem p, int[] tour)
    {
        double total = 0;
        for (int i = 0; i < tour.Length; i++) total += p.Dist![tour[i], tour[(i + 1) % tour.Length]];
        return total;
    }

    /// <summary>Number of diagonally attacking pairs; <c>perm[i]</c> is the row of the queen in column i.</summary>
    public static double NQueens(int[] perm)
    {
        int conflicts = 0;
        for (int i = 0; i < perm.Length; i++)
            for (int j = i + 1; j < perm.Length; j++)
                if (Math.Abs(perm[i] - perm[j]) == j - i) conflicts++;
        return conflicts;
    }
}
