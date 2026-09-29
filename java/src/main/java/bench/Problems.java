package bench;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Java ports of the reference fitness functions in {@code radiate_benchmarks/problems/}.
 * The aggregator re-scores every best solution with the Python reference, so these
 * must match it formula for formula.
 */
final class Problems {
    private Problems() {}

    enum Kind { CONTINUOUS, KNAPSACK, TSP, NQUEENS, MO }

    /** One problem instance, loaded from {@code spec/problems/<name>.json}. */
    record Problem(
        String name,
        Kind kind,
        String function,
        int size,
        int nObj,
        double lo,
        double hi,
        double[] weights,
        double[] values,
        double capacity,
        double penalty,
        double[][] dist
    ) {
        static Problem fromSpec(JsonNode spec) {
            String name = spec.get("name").asText();
            return switch (spec.get("kind").asText()) {
                case "continuous" -> new Problem(name, Kind.CONTINUOUS, spec.get("function").asText(),
                    spec.get("dim").asInt(), 1, spec.get("bounds").get(0).asDouble(),
                    spec.get("bounds").get(1).asDouble(), null, null, 0, 0, null);
                case "knapsack" -> new Problem(name, Kind.KNAPSACK, null, spec.get("n_items").asInt(), 1, 0, 0,
                    doubles(spec.get("weights")), doubles(spec.get("values")),
                    spec.get("capacity").asDouble(), spec.get("overweight_penalty").asDouble(), null);
                case "tsp" -> {
                    JsonNode coords = spec.get("coords");
                    int n = coords.size();
                    double[][] dist = new double[n][n];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            double dx = coords.get(i).get(0).asDouble() - coords.get(j).get(0).asDouble();
                            double dy = coords.get(i).get(1).asDouble() - coords.get(j).get(1).asDouble();
                            dist[i][j] = Math.sqrt(dx * dx + dy * dy);
                        }
                    }
                    yield new Problem(name, Kind.TSP, null, n, 1, 0, 0, null, null, 0, 0, dist);
                }
                case "nqueens" -> new Problem(name, Kind.NQUEENS, null, spec.get("n").asInt(), 1, 0, 0,
                    null, null, 0, 0, null);
                case "mo" -> new Problem(name, Kind.MO, spec.get("function").asText(), spec.get("n_var").asInt(),
                    spec.get("n_obj").asInt(), spec.get("bounds").get(0).asDouble(),
                    spec.get("bounds").get(1).asDouble(), null, null, 0, 0, null);
                default -> throw new IllegalArgumentException("unknown problem kind " + spec.get("kind"));
            };
        }

        private static double[] doubles(JsonNode array) {
            double[] out = new double[array.size()];
            for (int i = 0; i < out.length; i++) out[i] = array.get(i).asDouble();
            return out;
        }
    }

    static double continuous(String function, double[] x) {
        int n = x.length;
        switch (function) {
            case "sphere": {
                double s = 0;
                for (double v : x) s += v * v;
                return s;
            }
            case "rastrigin": {
                double s = 10.0 * n;
                for (double v : x) s += v * v - 10.0 * Math.cos(2 * Math.PI * v);
                return s;
            }
            case "rosenbrock": {
                double s = 0;
                for (int i = 0; i < n - 1; i++) {
                    s += 100.0 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
                }
                return s;
            }
            case "ackley": {
                double sum1 = 0, sum2 = 0;
                for (double v : x) {
                    sum1 += v * v;
                    sum2 += Math.cos(2 * Math.PI * v);
                }
                return -20 * Math.exp(-0.2 * Math.sqrt(sum1 / n)) - Math.exp(sum2 / n) + 20 + Math.E;
            }
            default:
                throw new IllegalArgumentException("unknown continuous function " + function);
        }
    }

    static double knapsack(Problem p, boolean[] bits) {
        double weight = 0, value = 0;
        for (int i = 0; i < bits.length; i++) {
            if (bits[i]) {
                weight += p.weights()[i];
                value += p.values()[i];
            }
        }
        return weight > p.capacity() ? value - p.penalty() * (weight - p.capacity()) : value;
    }

    /** Length of the closed tour (includes the leg from the last city back to the first). */
    static double tsp(Problem p, int[] tour) {
        double total = 0;
        for (int i = 0; i < tour.length; i++) total += p.dist()[tour[i]][tour[(i + 1) % tour.length]];
        return total;
    }

    /** Number of diagonally attacking pairs; {@code perm[i]} is the row of the queen in column i. */
    static double nqueens(int[] perm) {
        int conflicts = 0;
        for (int i = 0; i < perm.length; i++) {
            for (int j = i + 1; j < perm.length; j++) {
                if (Math.abs(perm[i] - perm[j]) == j - i) conflicts++;
            }
        }
        return conflicts;
    }

    static double[] multiObjective(String function, int nObj, double[] x) {
        switch (function) {
            case "zdt1":
            case "zdt3": {
                double f1 = x[0];
                double sum = 0;
                for (int i = 1; i < x.length; i++) sum += x[i];
                double g = 1.0 + 9.0 * sum / (x.length - 1);
                double h = 1.0 - Math.sqrt(f1 / g);
                if (function.equals("zdt3")) h -= (f1 / g) * Math.sin(10 * Math.PI * f1);
                return new double[] {f1, g * h};
            }
            case "dtlz2": {
                int m = nObj;
                double g = 0;
                for (int i = m - 1; i < x.length; i++) g += (x[i] - 0.5) * (x[i] - 0.5);
                double[] f = new double[m];
                for (int i = 0; i < m; i++) {
                    double val = 1.0 + g;
                    for (int j = 0; j < m - 1 - i; j++) val *= Math.cos(x[j] * Math.PI / 2.0);
                    if (i > 0) val *= Math.sin(x[m - 1 - i] * Math.PI / 2.0);
                    f[i] = val;
                }
                return f;
            }
            default:
                throw new IllegalArgumentException("unknown multi-objective function " + function);
        }
    }
}
