package bench;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAII;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.operator.crossover.impl.SBXCrossover;
import org.uma.jmetal.operator.mutation.impl.PolynomialMutation;
import org.uma.jmetal.problem.doubleproblem.impl.AbstractDoubleProblem;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

/**
 * jMetal's classic NSGA-II with SBX and polynomial mutation (eta=20), the same operators the
 * Python MO adapters use. jMetal is a multi-objective framework, so it only runs the MO suite.
 */
final class JMetalRunner implements Main.Library {

    @Override
    public String name() {
        return "jmetal";
    }

    @Override
    public boolean supports(Problems.Kind kind) {
        return kind == Problems.Kind.MO;
    }

    @Override
    public RunResult run(Problems.Problem p, Main.Config c, long seed) {
        JMetalRandom.getInstance().setSeed(seed);

        var problem = new SpecProblem(p);
        int pop = c.populationSize();
        // Initial population plus one offspring population per generation.
        int maxEvaluations = pop * (c.generations() + 1);
        NSGAII<DoubleSolution> algorithm = new NSGAIIBuilder<>(
                problem,
                new SBXCrossover(c.crossoverRate(), 20.0),
                // Per-variable probability, like DEAP's indpb and radiate's rate.
                new PolynomialMutation(c.mutationRate(), 20.0),
                pop)
            .setMaxEvaluations(maxEvaluations)
            .build();

        long t0 = System.nanoTime();
        algorithm.run();
        double wall = (System.nanoTime() - t0) / 1e9;

        List<DoubleSolution> front = SolutionListUtils.getNonDominatedSolutions(algorithm.result());
        List<double[]> objectives = new ArrayList<>(front.size());
        for (DoubleSolution s : front) objectives.add(s.objectives().clone());
        return RunResult.multi(objectives, wall);
    }

    /** A spec MO problem as a jMetal DoubleProblem (all objectives minimized, jMetal's convention). */
    private static final class SpecProblem extends AbstractDoubleProblem {
        private final Problems.Problem p;

        SpecProblem(Problems.Problem p) {
            this.p = p;
            name(p.name());
            numberOfObjectives(p.nObj());
            numberOfConstraints(0);
            variableBounds(
                new ArrayList<>(Collections.nCopies(p.size(), p.lo())),
                new ArrayList<>(Collections.nCopies(p.size(), p.hi())));
        }

        @Override
        public DoubleSolution evaluate(DoubleSolution solution) {
            double[] x = new double[p.size()];
            for (int i = 0; i < x.length; i++) x[i] = solution.variables().get(i);
            double[] f = Problems.multiObjective(p.function(), p.nObj(), x);
            System.arraycopy(f, 0, solution.objectives(), 0, f.length);
            return solution;
        }
    }
}
