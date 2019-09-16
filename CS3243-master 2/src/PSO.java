import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.text.DecimalFormat;

public class PSO {
    private static final int NUMOFPARTICLES = 25; // Set to a perfect square to make updating easier
    // The width of the particle mesh topology, should be the square root of NUMOFPARTICLES
    private static final int TOPOLOGYWIDTH = (int)Math.sqrt(NUMOFPARTICLES);
    private static final int NUMOFGAMES = 7;
    private static final int NUMOFITER = 30;

    private static final int NUMOFFEATURES = PlayerSkeleton.NUMOFFEATURES;
    private static final String PARTICLEWEIGHTSFN = "./resources/particleWeights.csv";
    private static final String OPTIMIZEDWEIGHTSFN = PlayerSkeleton.OPTIMIZEDWEIGHTSFN;

    public static final double INERTIA = 0.72;
    public static final double ACCELERATION1 = 1.42;
    public static final double ACCELERATION2 = 1.42;
    public static final double VMAX = 0.5; // Limits the velocity when updating the solution vectors

    public static final int NUMOFTHREADS = 10;

    public Particle[] particles;
    private final ExecutorService pool;

    /** Constructor for the particle swarm optimization class
     * Initializes all of the particles in the swarm with either predefined values or random variables
     *   if starting the optimization from scratch
     * @param particleSolutionVectors A predefined set of solution vectors to use for the swarm
     */
    public PSO() {
        Vector<double[]> particleSolutionVectors = readSolutionVectorsFromFile();
        particles = new Particle[NUMOFPARTICLES];
        // If we have a valid solution vector already specified
        if (particleSolutionVectors != null &&
                particleSolutionVectors.size() == NUMOFPARTICLES) {
            for (int i = 0; i < NUMOFPARTICLES; i++) {
                particles[i] = new Particle(NUMOFGAMES, NUMOFFEATURES);
                particles[i].neighbours = findNeighbours(i).clone();
                particles[i].setSolutionVector(particleSolutionVectors.elementAt(i));
            }
        } else {
            Random rand = new Random();
            // Initialize all of the particles that will be used by PSO
            for (int i = 0; i < NUMOFPARTICLES; i++) {
                particles[i] = new Particle(NUMOFGAMES, NUMOFFEATURES);
                // Initialize the neighbours for each particle to avoid calculating them each loop later
                particles[i].neighbours = findNeighbours(i).clone();
                // randomly generate the solution vector
                double[] sv = new double[NUMOFFEATURES];
                // Generate a random number in [-1, 1) for each solution vector value
                for (int j = 0; j < NUMOFFEATURES; j++) {
                    sv[j] = rand.nextDouble()*2 - 1;
                }
                particles[i].setSolutionVector(sv);
            }
        }

        // Initialize a thread pool that creates threads as needed
        pool = Executors.newFixedThreadPool(NUMOFTHREADS);
    }

    /** Executes the particle swarm optimization algorithm
     *  @author Josh
     */
    public void runPSO() {
        System.out.println("Starting, hold on to your hats....");
        long start = System.currentTimeMillis();

        // Run the optimization algorithm NUMOFITER times
        for (int i = 0; i < NUMOFITER; i++) {
            Collection<Callable<Double>> tasks = new ArrayList<Callable<Double>>();
            // Loop through all particles
            for (int particle = 0; particle < NUMOFPARTICLES; particle++) {
                // Now run a preset number of games to get a fair fitness value
                for (int game = 0; game < NUMOFGAMES; game++) {
                   tasks.add(new Handler(particles[particle]));
                }
            }

            // Run all the games for the current iteration then save the values to their respective particle
            try {
                List<Future<Double>> futures = pool.invokeAll(tasks);
                for (int particle = 0; particle < NUMOFPARTICLES; particle++) {
                    // Check to make sure every game finished, and if it did, save the results
                    for (int game = 0; game < NUMOFGAMES; game++) {
                       int index = particle*NUMOFGAMES + game;
                       if (futures.get(index).isDone()) {
                           particles[particle].addFitnessValue(futures.get(index).get(), game);
                       }
                    }
                    // Once all values are saved, calculate the average fitness for that particle
                    particles[particle].calculateAverageFitness();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }


            // Output the results of each particle to determine if learning is occurring
            System.out.println("Fitness scores after " + String.valueOf(i + 1) + " iteration(s):");
            DecimalFormat df2 = new DecimalFormat(".##");
            for (int j = 0; j < NUMOFPARTICLES; j++) {
                System.out.println("Particle " + j + ": " +
                                    String.valueOf(df2.format(particles[j].getFitness())));
            }
            System.out.println("");

            updateParticles();

            tasks.clear();
        }
        writeSolutionVectorsToFile();

        // Find the best fitness value to be used throughout the entire training session
        int bestParticleIndex = 0;
        double bestParticleFitnessScore = particles[0].getBestFitness();
        for (int i = 1; i < particles.length; i++) {
            if (particles[i].getBestFitness() > bestParticleFitnessScore) {
                bestParticleIndex = i;
                bestParticleFitnessScore = particles[i].getBestFitness();
            }
        }

        final double[] bestSolutionVector = particles[bestParticleIndex].getBestSolutionVector();
        writeBestSolutionVectorToFile(bestSolutionVector, bestParticleFitnessScore);
        long stop = System.currentTimeMillis();
        long dif = stop - start;
        System.out.println("Wow! That took " + String.valueOf((int)dif/1000) + " seconds");

        pool.shutdownNow();
    }

    class Handler implements Callable<Double> {
        private Particle particle;
        /*
         * pileHeight [0], max [1]
         * holes [2], max [3]
         * row transitions [4], max [5]
         * col transitions [6], max [7]
         * turns [8]
         */
        public int[] utilityValues;

        Handler(Particle particle) {
            this.particle = particle;
            this.utilityValues = new int[9]; // Initialized to zero
        }

        public Double call() {
            PlayerSkeleton p = new PlayerSkeleton();
            State s = new State();
            p.setWeights(particle.getSolutionVector());
            // Reset all the values for the
            while(!s.hasLost()) {
                s.makeMove(p.pickMove(s,s.legalMoves(), utilityValues));
            }
            return PSO.calculateFitnessForGame(utilityValues, s.getRowsCleared());
        }
      }


    /** Calculate the fitness for a game played by the particle based on its performance
     *   Performance is based on the number of lines cleared, the max/average height while playing,
     *   the max/average number of holes, the max/average number of row transitions, and the max/average
     *   number of column transitions
     *  This fitness score is used to help differentiate scores early in the optimization process, where often
     *   many nodes will obtain roughly the same number of lines (but one might be playing better)
     * @param uv The utility values gathered throughout the game
     * @param rowsCleared The number of rows cleared during the game
     * @return The calculated fitness value based on the input variables
     */
    public static double calculateFitnessForGame(int[] uv, int rowsCleared) {
        double fitness = rowsCleared;
        // (Max - average)/max * 500
        // Pile Height
        if (uv[1] != 0) {
            fitness += ((double) uv[1] - ((double)uv[0] / (double) uv[8])) / uv[1] * 500;

        }
        // Holes
        if (uv[3] != 0) {
            fitness += ((double) uv[3] - ((double)uv[2] / (double)uv[8])) / uv[3] * 500;

        }
        // Row transitions
        if (uv[5] != 0) {
            fitness += ((double) uv[5] - ((double)uv[4] / (double) uv[8])) / uv[5] * 500;

        }
        // Col Transitions
        if (uv[7] != 0) {
            fitness += ((double) uv[7] - ((double)uv[6] / (double)uv[8])) / uv[7] * 500;

        }
        //System.out.print(fitness + "\n");
        return rowsCleared;
    }

    /** Updates all particles in the array to their new position based on the current position and
     *      the particles that neighbour it.
 *      @author Josh
     */
    public void updateParticles() {
        Random rand = new Random();
        double r1 = rand.nextDouble();
        double r2 = rand.nextDouble();

        for (int i = 0; i < NUMOFPARTICLES; i++) {
            // For the current particle being evaluated
            double[] bestSolution = particles[i].getBestSolutionVector();
            double[] currentSolution = particles[i].getSolutionVector();

            // Find the best solution vector found by any neighbouring particle
            double bestNeighbouringFitness = particles[particles[i].neighbours[0]].getBestFitness();
            int bestNeighbourIndex = particles[i].neighbours[0];
            for (int j = 1; j < particles[i].neighbours.length; j++) {
                if (particles[particles[i].neighbours[j]].getBestFitness() > bestNeighbouringFitness) {
                    bestNeighbouringFitness = particles[particles[i].neighbours[j]].getBestFitness();
                    bestNeighbourIndex = particles[i].neighbours[j];
                }
            }
            double[] bestNeighbouringSolutionVector = particles[bestNeighbourIndex].getBestSolutionVector();

            // x(t+1) = x(t) + v(t+1), new position = old position + updated velocity vector
            // v(t+1) = phi * v(t) + c1*r1(t)*(y(t)-x(t))+c2*r2(t)*(yhat(t)-x(t))
                // phi = INERTIA; c1, c2 = ACCELERATION1,2; r1, r2 are random numbers in (0, 1)
            double[] newVelocity = particles[i].getVelocity().clone();
            for (int j = 0; j < NUMOFFEATURES; j++) {
                newVelocity[j] = newVelocity[j]*INERTIA +
                        ACCELERATION1*r1*(bestSolution[j] - currentSolution[j]) +  // Cognitive component
                        ACCELERATION2*r2*(bestNeighbouringSolutionVector[j] - currentSolution[j]); // Social component
            }

            double[] newSolution = particles[i].getSolutionVector().clone();
            for (int j = 0; j < NUMOFFEATURES; j++) {
                // Limit the velocity to prevent a particle from possibly stepping over a more optimal solution
                if (newVelocity[j] > VMAX) {
                    newSolution[j] += VMAX;
                } else if (newVelocity[j] < -VMAX) {
                    newSolution[j] -= VMAX;
                } else {
                    newSolution[j] += newVelocity[j];
                }
            }

            // Now update the particle with its new solution vector and velocity
            particles[i].setSolutionVector(newSolution);
            particles[i].setVelocity(newVelocity);
        }
    }

    /** Finds the position of neighbouring particles in the mesh topology
     *      A particle's neighbours are those situated to the N, E, S, W in a 2D grid
     *      Particles in the corners and edges have fewer neighbours
     * @param index The index of the particle being referenced
     * @return A list of neighbouring particle indices in the "particles" list
     */
    public int[] findNeighbours(int index) {
        int[] neighbours;
        if (NUMOFPARTICLES == 1) {
            neighbours = new int[0];
            return neighbours;
        }

        if (index == 0) {
            // If the particle is in the top-left corner
            neighbours = new int[2];
            neighbours[0] = 1;
            neighbours[1] = TOPOLOGYWIDTH;
        } else if (index == TOPOLOGYWIDTH-1) {
            // Top-right
            neighbours = new int[2];
            neighbours[0] = TOPOLOGYWIDTH - 2;
            neighbours[1] = index + TOPOLOGYWIDTH;
        } else if (index == NUMOFPARTICLES - TOPOLOGYWIDTH) {
            // Bottom-left
            neighbours = new int[2];
            neighbours[0] = NUMOFPARTICLES - TOPOLOGYWIDTH + 1;
            neighbours[1] = index - TOPOLOGYWIDTH;
        } else if (index == NUMOFPARTICLES - 1) {
            // Bottom-right
            neighbours = new int[2];
            neighbours[0] = NUMOFPARTICLES - 1;
            neighbours[1] = index - TOPOLOGYWIDTH;
        } else if (index < TOPOLOGYWIDTH) {
            // Top row
            neighbours = new int[3];
            neighbours[0] = index - 1;
            neighbours[1] = index + 1;
            neighbours[2] = index + TOPOLOGYWIDTH;
        } else if (index > NUMOFPARTICLES - TOPOLOGYWIDTH) {
            // Bottom row
            neighbours = new int[3];
            neighbours[0] = index - 1;
            neighbours[1] = index + 1;
            neighbours[2] = index - TOPOLOGYWIDTH;
        } else if (index % TOPOLOGYWIDTH == 0) {
            // Left column
            neighbours = new int[3];
            neighbours[0] = index - TOPOLOGYWIDTH;
            neighbours[1] = index + TOPOLOGYWIDTH;
            neighbours[2] = index + 1;
        } else if ((index + 1) % TOPOLOGYWIDTH == 0) {
            // Right column
            neighbours = new int[3];
            neighbours[0] = index - TOPOLOGYWIDTH;
            neighbours[1] = index + TOPOLOGYWIDTH;
            neighbours[2] = index - 1;
        } else {
            // Middle of mesh (not corner or edge)
            neighbours = new int[4];
            neighbours[0] = index - TOPOLOGYWIDTH;
            neighbours[1] = index + TOPOLOGYWIDTH;
            neighbours[2] = index - 1;
            neighbours[3] = index + 1;
        }
        return neighbours;
    }

    public Vector<double[]> readSolutionVectorsFromFile() {
        BufferedReader reader;
        Vector<double[]> particleSolutionVectors = new Vector<double[]>();
        try {
            // Optimized weights will contain the best weights found by PSO previously
            reader = new BufferedReader(new FileReader(PARTICLEWEIGHTSFN));
            String line = reader.readLine();
            String[] splitString;
            while (line != null) {
                splitString = line.split(",");
                // If there are not enough weights for the number of features being used,
                //   then we cannot use these weights
                if (splitString.length != NUMOFFEATURES) {
                    particleSolutionVectors = null;
                    break;
                }
                double[] particleSolutionVector = new double[splitString.length];
                for (int i = 0; i < splitString.length; i++) {
                    particleSolutionVector[i] = Double.parseDouble(splitString[i]);
                }
                particleSolutionVectors.add(particleSolutionVector);
                line = reader.readLine();
            }
            reader.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        return particleSolutionVectors;
    }

    /** Writes the final values for the solution vector of each particle to the file
     */
    public void writeSolutionVectorsToFile() {
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(PARTICLEWEIGHTSFN);
            bw = new BufferedWriter(fw);
            String data = "";
            for (int i = 0; i < particles.length; i++) {
                data = data + String.join(",", particles[i].getSolutionVectorAsString()) + "\n";
            }
            bw.write(data);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) bw.close();
                if (fw != null) fw.close();
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }
    }

    /** Writes the best solution vector found during execution to the particleWeights file
     *      This will be used in the non-learning version of the game
     *      Also outputs the results into the console for easy reading
     * @param sv The best solution vector gathered from all particles
     */
    public void writeBestSolutionVectorToFile(double[] sv, double fitnessScore) {
        String vectorAsString = "";
        String value = "";
        System.out.println("The best fitness score achieved is " + String.valueOf(fitnessScore));
        // Print the result to console for easy viewing
        System.out.println("The weights used were:");
        for (int i = 0; i < sv.length; i++) {
            value = String.valueOf(sv[i]);
            if (i != sv.length - 1) {
                System.out.print(value + ", ");
            } else {
                System.out.print(value + "\n");
            }
            vectorAsString +=  value + "\n";
        }

        // Now write the weights to the file
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(OPTIMIZEDWEIGHTSFN);
            bw = new BufferedWriter(fw);
            bw.write(vectorAsString);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) bw.close();
                if (fw != null) fw.close();
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }
    }
}
