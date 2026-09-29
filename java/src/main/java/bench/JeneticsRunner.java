package bench;

import io.jenetics.BitChromosome;
import io.jenetics.BitGene;
import io.jenetics.DoubleGene;
import io.jenetics.EliteSelector;
import io.jenetics.EnumGene;
import io.jenetics.Gene;
import io.jenetics.MeanAlterer;
import io.jenetics.Genotype;
import io.jenetics.Mutator;
import io.jenetics.PartiallyMatchedCrossover;
import io.jenetics.Phenotype;
import io.jenetics.SwapMutator;
import io.jenetics.TournamentSelector;
import io.jenetics.UniformCrossover;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.Limits;
import io.jenetics.ext.SimulatedBinaryCrossover;
import io.jenetics.ext.moea.MOEA;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.UFTournamentSelector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.util.DoubleRange;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.RandomRegistry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGeneratorFactory;

/**
 * Jenetics, configured the way its documentation does it for each representation.
 *
 * <p>Evaluation runs on the calling thread ({@code executor(Runnable::run)}): Jenetics
 * evaluates fitness on a ForkJoinPool by default, which would give it every core while
 * every other library here runs single-threaded.
 */
final class JeneticsRunner implements Main.Library {

    @Override
    public String name() {
        return "jenetics";
    }

    @Override
    public boolean supports(Problems.Kind kind) {
        return true;
    }

    @Override
    public RunResult run(Problems.Problem p, Main.Config c, long seed) {
        var random = RandomGeneratorFactory.of("L64X256MixRandom").create(seed);
        return RandomRegistry.with(random).call(() -> runSeeded(p, c));
    }

    private static RunResult runSeeded(Problems.Problem p, Main.Config c) {
        switch (p.kind()) {
            case CONTINUOUS: {
                Codec<double[], DoubleGene> codec = Codecs.ofVector(new DoubleRange(p.lo(), p.hi()), p.size());
                Engine<DoubleGene, Double> engine = Engine
                    .builder((double[] x) -> Problems.continuous(p.function(), x), codec)
                    .minimizing()
                    .populationSize(c.populationSize())
                    .executor(Runnable::run)
                    .offspringSelector(new TournamentSelector<>(3))
                    .survivorsSelector(new EliteSelector<>(1))
                    // MeanAlterer + Mutator, as in Jenetics' own Rastrigin example. (GaussianMutator
                    // uses sigma = range/4, too coarse to converge on these bounds.)
                    .alterers(new MeanAlterer<>(c.crossoverRate()), new Mutator<>(c.mutationRate()))
                    .build();
                return single(engine, codec, c.generations(), true,
                    x -> Arrays.stream(x).mapToObj(Double::toString).toList());
            }
            case KNAPSACK: {
                int n = p.size();
                Codec<boolean[], BitGene> codec = Codec.of(
                    Genotype.of(BitChromosome.of(n, 0.5)),
                    gt -> {
                        BitChromosome ch = gt.chromosome().as(BitChromosome.class);
                        boolean[] bits = new boolean[n];
                        for (int i = 0; i < n; i++) bits[i] = ch.booleanValue(i);
                        return bits;
                    });
                Engine<BitGene, Double> engine = Engine
                    .builder((boolean[] bits) -> Problems.knapsack(p, bits), codec)
                    .populationSize(c.populationSize())
                    .executor(Runnable::run)
                    .offspringSelector(new TournamentSelector<>(3))
                    .survivorsSelector(new EliteSelector<>(1))
                    .alterers(new UniformCrossover<>(c.crossoverRate()), new Mutator<>(c.mutationRate()))
                    .build();
                return single(engine, codec, c.generations(), false, bits -> {
                    List<String> out = new ArrayList<>(bits.length);
                    for (boolean b : bits) out.add(b ? "1" : "0");
                    return out;
                });
            }
            case TSP:
            case NQUEENS: {
                Codec<int[], EnumGene<Integer>> codec = Codecs.ofPermutation(p.size());
                Engine<EnumGene<Integer>, Double> engine = Engine
                    .builder((int[] perm) -> p.kind() == Problems.Kind.TSP
                        ? Problems.tsp(p, perm) : Problems.nqueens(perm), codec)
                    .minimizing()
                    .populationSize(c.populationSize())
                    .executor(Runnable::run)
                    .offspringSelector(new TournamentSelector<>(3))
                    .survivorsSelector(new EliteSelector<>(1))
                    .alterers(new PartiallyMatchedCrossover<>(c.crossoverRate()), new SwapMutator<>(c.mutationRate()))
                    .build();
                return single(engine, codec, c.generations(), true,
                    perm -> Arrays.stream(perm).mapToObj(Integer::toString).toList());
            }
            case MO: {
                Codec<double[], DoubleGene> codec = Codecs.ofVector(new DoubleRange(p.lo(), p.hi()), p.size());
                // NSGA-II: unique-fitness crowded tournament for parents, rank + crowding for
                // survivors. Jenetics has SBX but no polynomial mutation, so its uniform Mutator.
                Engine<DoubleGene, Vec<double[]>> engine = Engine
                    .builder((double[] x) -> Vec.of(Problems.multiObjective(p.function(), p.nObj(), x)), codec)
                    .minimizing()
                    .populationSize(c.populationSize())
                    .executor(Runnable::run)
                    .offspringFraction(0.5)
                    .offspringSelector(UFTournamentSelector.ofVec())
                    .survivorsSelector(NSGA2Selector.ofVec())
                    .alterers(new SimulatedBinaryCrossover<>(c.crossoverRate(), 20.0), new Mutator<>(c.mutationRate()))
                    .build();

                long t0 = System.nanoTime();
                ISeq<Phenotype<DoubleGene, Vec<double[]>>> front = engine.stream()
                    .limit(Limits.byFixedGeneration(c.generations()))
                    .collect(MOEA.toParetoSet(new IntRange(c.populationSize(), c.populationSize() + 50)));
                double wall = (System.nanoTime() - t0) / 1e9;

                List<double[]> objectives = front.stream().map(ph -> ph.fitness().data()).toList();
                return RunResult.multi(objectives, wall);
            }
            default:
                throw new IllegalStateException();
        }
    }

    /** Runs a single-objective engine; the timer covers only the evolution stream. */
    private static <T, G extends Gene<?, G>> RunResult single(
        Engine<G, Double> engine,
        Codec<T, G> codec,
        int generations,
        boolean minimize,
        java.util.function.Function<T, List<String>> solution
    ) {
        List<Double> history = new ArrayList<>(generations);
        double[] best = {minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY};

        long t0 = System.nanoTime();
        EvolutionResult<G, Double> last = engine.stream()
            .limit(Limits.byFixedGeneration(generations))
            .peek(r -> {
                best[0] = minimize ? Math.min(best[0], r.bestFitness()) : Math.max(best[0], r.bestFitness());
                history.add(best[0]);
            })
            .collect(EvolutionResult.toBestEvolutionResult());
        double wall = (System.nanoTime() - t0) / 1e9;

        Phenotype<G, Double> bestPhenotype = last.bestPhenotype();
        return RunResult.single(
            bestPhenotype.fitness(), solution.apply(codec.decode(bestPhenotype.genotype())), history, wall);
    }
}
