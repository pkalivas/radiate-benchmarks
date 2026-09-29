package bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Java language runner: Jenetics and jMetal x every problem in the spec x every seed.
 *
 * <p>Invoked by {@code java/run.sh}; reads {@code --spec} and writes {@code runs.csv},
 * {@code history.csv} and {@code fronts.csv} to {@code --out} as described in schema.md.
 * Before the timed seeds of each (library, problem), one untimed warm-up run lets the JIT
 * compile the hot paths, so JIT time doesn't land in the first seed's wall_time_s.
 */
public final class Main {
    static final String LANGUAGE = "java";
    static final long WARMUP_SEED = 0;

    record Config(
        int populationSize,
        int generations,
        double crossoverRate,
        double mutationRate,
        List<Long> seeds,
        List<String> problems,
        String specHash
    ) {}

    interface Library {
        String name();

        boolean supports(Problems.Kind kind);

        RunResult run(Problems.Problem problem, Config config, long seed);
    }

    public static void main(String[] args) throws IOException {
        Path specDir = null;
        Path outDir = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--spec" -> specDir = Path.of(args[++i]);
                case "--out" -> outDir = Path.of(args[++i]);
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        if (specDir == null || outDir == null) throw new IllegalArgumentException("--spec and --out are required");
        Files.createDirectories(outDir);

        ObjectMapper json = new ObjectMapper();
        JsonNode raw = json.readTree(specDir.resolve("config.json").toFile());
        List<Long> seeds = new ArrayList<>();
        raw.get("seeds").forEach(s -> seeds.add(s.asLong()));
        List<String> problemNames = new ArrayList<>();
        raw.get("problems").forEach(s -> problemNames.add(s.asText()));
        Config config = new Config(
            raw.get("population_size").asInt(),
            raw.get("generations").asInt(),
            raw.get("crossover_rate").asDouble(),
            raw.get("mutation_rate").asDouble(),
            seeds,
            problemNames,
            raw.get("spec_hash").asText());

        Properties versions = new Properties();
        try (InputStream in = Main.class.getResourceAsStream("/versions.properties")) {
            versions.load(in);
        }

        List<Library> libraries = List.of(new JeneticsRunner(), new JMetalRunner());
        List<String> runs = new ArrayList<>();
        List<String> history = new ArrayList<>();
        List<String> fronts = new ArrayList<>();

        for (String name : config.problems()) {
            Problems.Problem problem = Problems.Problem.fromSpec(
                json.readTree(specDir.resolve("problems").resolve(name + ".json").toFile()));
            for (Library lib : libraries) {
                if (!lib.supports(problem.kind())) {
                    System.out.printf("%12s | %8s | not supported, skipping%n", problem.name(), lib.name());
                    continue;
                }
                lib.run(problem, config, WARMUP_SEED);

                for (long seed : config.seeds()) {
                    System.out.printf("%12s | %8s | seed=%d ", problem.name(), lib.name(), seed);
                    RunResult r = lib.run(problem, config, seed);

                    String key = String.join(",", LANGUAGE, lib.name(), problem.name(), Long.toString(seed));
                    runs.add(String.join(",",
                        LANGUAGE,
                        lib.name(),
                        versions.getProperty(lib.name()),
                        problem.name(),
                        Long.toString(seed),
                        r.bestFitness() == null ? "" : Double.toString(r.bestFitness()),
                        String.join(";", r.bestSolution()),
                        Double.toString(r.wallTimeS()),
                        config.specHash(),
                        Instant.now().truncatedTo(ChronoUnit.SECONDS).toString()));
                    for (int g = 0; g < r.history().size(); g++) {
                        history.add(key + "," + g + "," + r.history().get(g));
                    }
                    for (int i = 0; i < r.front().size(); i++) {
                        StringBuilder objectives = new StringBuilder();
                        for (double v : r.front().get(i)) {
                            if (!objectives.isEmpty()) objectives.append(';');
                            objectives.append(v);
                        }
                        fronts.add(key + "," + i + "," + objectives);
                    }

                    if (r.bestFitness() == null) {
                        System.out.printf("-> front  time=%.3fs%n", r.wallTimeS());
                    } else {
                        System.out.printf("-> best=%.4f  time=%.3fs%n", r.bestFitness(), r.wallTimeS());
                    }
                }
            }
        }

        write(outDir.resolve("runs.csv"),
            "language,library,library_version,problem,seed,best_fitness,best_solution,wall_time_s,spec_hash,timestamp", runs);
        write(outDir.resolve("history.csv"), "language,library,problem,seed,generation,best_so_far", history);
        write(outDir.resolve("fronts.csv"), "language,library,problem,seed,point,objectives", fronts);
        System.out.printf("wrote %d runs to %s/%n", runs.size(), outDir);
    }

    private static void write(Path path, String header, List<String> rows) throws IOException {
        StringBuilder text = new StringBuilder(header).append('\n');
        for (String row : rows) text.append(row).append('\n');
        Files.writeString(path, text);
    }
}
