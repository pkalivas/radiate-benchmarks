package bench;

import java.util.List;

/**
 * One finished run. {@code bestFitness} is null for multi-objective runs, whose score
 * (hypervolume) is computed by the aggregator from {@code front}.
 */
record RunResult(
    Double bestFitness,
    List<String> bestSolution,
    List<Double> history,
    double wallTimeS,
    List<double[]> front
) {
    static RunResult single(double best, List<String> solution, List<Double> history, double wall) {
        return new RunResult(best, solution, history, wall, List.of());
    }

    static RunResult multi(List<double[]> front, double wall) {
        return new RunResult(null, List.of(), List.of(), wall, front);
    }
}
