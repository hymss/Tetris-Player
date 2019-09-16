import java.util.Arrays;

public class Particle {
    private double[] fitnessValues;
    private double[] solutionVector;
    private double[] bestSolutionVector; // The solution vector that results in the best fitness value
    private double fitness;
    private double bestFitness; // The best fitness value obtained while training
    private double[] velocity;
    
    public int[] neighbours;
    
    public Particle(int numOfGames, int numOfFeatures) {
        fitnessValues = new double[numOfGames];
        solutionVector = new double[numOfFeatures];
        bestSolutionVector = new double[numOfFeatures];
        bestFitness = Double.NEGATIVE_INFINITY;
        velocity = new double[numOfFeatures];
    }
    
    public void setSolutionVector(double[] sv) {
        solutionVector = sv.clone();
    }
    
    public double[] getSolutionVector() {
        return solutionVector;
    }
    
    public String[] getSolutionVectorAsString() {
        String[] s = new String[solutionVector.length];
        for (int i = 0; i < solutionVector.length; i++) {
            s[i] = String.valueOf(solutionVector[i]);
        }
        return s;
    }
    
    public double[] getBestSolutionVector() {
        return bestSolutionVector;
    }
    
    public double[] getVelocity() {
        return velocity;
    }
    
    public void setVelocity(double[] vel) {
        velocity = vel.clone();
    }
    
    public void addFitnessValue(double fitness, int index) {
        fitnessValues[index] = fitness;
    }
    
    public void calculateAverageFitness() {
        Arrays.sort(fitnessValues);
        if (fitnessValues.length % 2 == 0) {
            fitness = (fitnessValues[fitnessValues.length / 2 - 1] + 
                    fitnessValues[fitnessValues.length / 2]) / 2;
        } else {
            fitness = fitnessValues[fitnessValues.length / 2 - 1];
        }
        
        // Check to see if the newest fitness is the best one so far
        if (fitness > bestFitness) {
            bestFitness = fitness;
            bestSolutionVector = solutionVector.clone();
        }
    }
    
    public double getFitness() {
        return fitness;
    }
    
    public double getBestFitness() {
        return bestFitness;
    }
}
