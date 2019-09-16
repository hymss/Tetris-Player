import java.io.BufferedReader;
import java.io.FileReader;

public class PlayerSkeleton {
    private static int[][][] pBottom = State.getpBottom();
    private static int[][][] pTop = State.getpTop();
    private static int[][] pHeight = State.getpWidth();
    private static int[][] pWidth = State.getpHeight();

    public static final String OPTIMIZEDWEIGHTSFN = "./resources/optimizedWeights.csv";
    public static final int NUMOFFEATURES = 13;
    
    public static final int LINESCLEAREDINDEX = 0;
    //public static final int HEIGHTINDEX= 1;
    //public static final int BUMPINESSINDEX = 2;
    public static final int HOLESINDEX = 1;
    public static final int PILEHEIGHTINDEX = 2;
    public static final int CONHOLESINDEX = 3; // Connected holes
    public static final int ALTITUDEDIFINDEX = 4; // Altitude difference
    public static final int SUMOFWELLDEPTHSINDEX = 5;
    public static final int MAXWELLDEPTHINDEX = 6;
    public static final int LANDINGHEIGHTINDEX = 7;
    public static final int BLOCKCOUNTINDEX = 8;
    public static final int WBLOCKCOUNTINDEX = 9; // Weighted block count
    public static final int ROWTRANSINDEX = 10; // Row transition
    public static final int COLTRANSINDEX = 11; // Column transition
    public static final int ERODEDBLOCKINDEX = 12;
    
    
    public static final boolean RUNPARTICLESWARM = false;
    public static final boolean DRAWBOARD = false;
    
    private double[] weights;
    
    /** Used to encapsulate the new fields generated by pickMove
     * 
     * @author Josh
     *
     */
    public static class NewField {
        public int[][] field;
        public int[] top;
        public int height;
        public int turn;
    }
    
    public PlayerSkeleton() {
        pBottom = State.getpBottom();
        pTop = State.getpTop();
        pWidth = State.getpWidth();
        pHeight = State.getpHeight();
    }
    
    public static void main(String[] args) {
        
        if (RUNPARTICLESWARM) {
            PSO swarm = new PSO();
            swarm.runPSO();
            return;
        } else {
            State s = new State();
            if (DRAWBOARD) {
                new TFrame(s);
            }
            PlayerSkeleton p = new PlayerSkeleton();
            
            // Retrieve the weights from an external file and stores them in weights
            BufferedReader reader;
            double[] inputWeights = new double[NUMOFFEATURES];
            int i = 0;
            try {
                reader = new BufferedReader(new FileReader(OPTIMIZEDWEIGHTSFN));
                String line = reader.readLine();
                while (line != null && i < NUMOFFEATURES) {
                    try {
                        inputWeights[i] = Double.parseDouble(line);
                    } catch (NumberFormatException e) {
                        e.printStackTrace();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    line = reader.readLine();
                    i++;
                }
                reader.close();
                
            } catch (Exception e) {
                e.printStackTrace();
            }
            
            p.setWeights(inputWeights);
            System.out.println("Starting...");
            int linesCleared = 10000;
            // Start the game
            while(!s.hasLost()) {
                s.makeMove(p.pickMove(s,s.legalMoves()));
                if (DRAWBOARD) {
                    s.draw();   
                    s.drawNext(0,0);
                    try {
                        Thread.sleep(1);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                if (s.getRowsCleared() >= linesCleared) {
                    System.out.println(s.getRowsCleared() + " lines cleared");
                    linesCleared += 10000;
                }
            }
            System.out.println("You have completed "+s.getRowsCleared()+" rows.");
        }
    }
    
    /** Overloaded pick move for particleSwarm
     * @param s Current state
     * @param legalMoves all valid legal moves
     * @param utilityValues a reference to the values used in calculating fitness for the current game
     * @return selected move with best utility
     */
    public int pickMove(State s, int[][] legalMoves, int[] utilityValues) {
        NewField updatedField;
        int orient;
        int slot;
        double[] maxUtil = new double[5];
        maxUtil[0] = Double.NEGATIVE_INFINITY;
        int bestMove = 0;        
        
        // Loop through each possible move and choose the best one
        for (int moveIndex = 0; moveIndex < legalMoves.length; moveIndex++) {
            orient = legalMoves[moveIndex][State.ORIENT];
            slot = legalMoves[moveIndex][State.SLOT];
            
            // Calculate the updated game board
            updatedField = updateField(s, orient, slot);
            // Causes the game to lose, so try the next move
            if (updatedField == null) {
                continue;
            }
            
            //Check if the new game board is the best one so far
            double[] util = calculateUtility_PSO(updatedField).clone();
            if (maxUtil[0] < util[0]) {
                maxUtil = util.clone();
                bestMove = moveIndex;
            }
        }
        
        /*
         *  We need to keep track of the performance of the algorithm at each turn
         *  The best algorithms are those that minimize these values
         *  The results are reflected in the fitness value at the end of the game
         */
        
        // Pile Height
        utilityValues[0] += (int)maxUtil[1];
        if (utilityValues[1] < maxUtil[1]) {
            utilityValues[1] = (int)maxUtil[1];
        }
        // Holes
        utilityValues[2] += (int)maxUtil[2];
        if (utilityValues[3] < maxUtil[2]) {
            utilityValues[3] = (int)maxUtil[2];
        }
        // Row transitions
        utilityValues[4] += (int)maxUtil[3];
        if (utilityValues[5] < maxUtil[3]) {
            utilityValues[5] = (int)maxUtil[3];
        }
        // Col Transitions
        utilityValues[6] += (int)maxUtil[4];
        if (utilityValues[7] < maxUtil[4]) {
            utilityValues[7] = (int)maxUtil[4];
        }
        // Number of moves made
        utilityValues[8]++;
        
        return bestMove;
    }
    
    //implement this function to have a working system
    public int pickMove(State s, int[][] legalMoves) {
        NewField updatedField;
        int orient;
        int slot;
        double maxUtil = Double.NEGATIVE_INFINITY;
        int bestMove = 0;        
        
        // Loop through each possible move and 
        for (int moveIndex = 0; moveIndex < legalMoves.length; moveIndex++) {
            orient = legalMoves[moveIndex][State.ORIENT];
            slot = legalMoves[moveIndex][State.SLOT];
            
            // Calculate the updated game board
            updatedField = updateField(s, orient, slot);
            // Causes the game to lose, so try the next move
            if (updatedField == null) {
                continue;
            }
            
            //Check if the new game board is the best one so far
            double util = calculateUtility(updatedField);
            if (maxUtil < util) {
                maxUtil = util;
                bestMove = moveIndex;
            }
        }
        
        return bestMove;
    }
    
    /** Updates the current field state to one where the current piece is placed
     * @author JoshMcManus
     * @param s The current state
     * @param orient The orientation of the next piece to place
     * @param slot The position on the board to place the piece
     * @return An updated board containing the new piece, null if the move causes a game over
     */
    public NewField updateField(State s, int orient, int slot) {
        int nextPiece = s.nextPiece;
        int turn = s.getTurnNumber();
        turn++;
        
        NewField nf = new NewField();
        nf.turn = turn;
        nf.top = new int[State.COLS];
        nf.top = s.getTop().clone();
        int[][] fieldRef = s.getField();
        nf.field = new int[fieldRef.length][];
        
        // Copy over the current field
        for(int i = 0; i < fieldRef.length; i++) {
            nf.field[i] = fieldRef[i].clone();
        }
        
        
        //////////////////////////////////////////////////////////
        // Modified version of the State.java makeMove function //
        //////////////////////////////////////////////////////////
        
        // height if the first column makes contact
        nf.height = nf.top[slot]-pBottom[nextPiece][orient][0];
        // for each column beyond the first in the piece
        for(int c = 1; c < pWidth[nextPiece][orient];c++) {
            nf.height = Math.max(nf.height,nf.top[slot+c]-pBottom[nextPiece][orient][c]);
        }
        
        // check if game ended
        if(nf.height+pHeight[nextPiece][orient] >= State.ROWS) {
            return null;
        }
        
      // for each column in the piece - fill in the appropriate blocks
        for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
            
            //from bottom to top of brick
            for(int h = nf.height+pBottom[nextPiece][orient][i]; h < nf.height+pTop[nextPiece][orient][i]; h++) {
                nf.field[h][i+slot] = turn;
            }
        }
        
      //adjust top
        for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
            nf.top[slot+c]=nf.height+pTop[nextPiece][orient][c];
        }
        
        return nf;

    }
    
    /** Calculates the utility for a particular move
     * @author JoshMcManus
     * @param s The current state of the Tetris board
     * @return The utility value for the current game board
     */
    public double calculateUtility(NewField f) {
        double util = 0;
        
        /* To calculate the utility we will use the following criteria:
         * 1) Complete Lines
         * 2) Aggregate Height
         * 3) Bumpiness
         * 4) Holes
         * 5) Pile Height
         * 6) Connected Holes
         * 7) Altitude Difference
         * 8) Sum of well depths
         * 9) Max well depth
         */
        int[] wells = sumAndMaxOfWellDepths(f);
        int[] blocks = blockCount_Weighted(f);
        int[] rows_eroded = linesCleared(f);
        util =  weights[LINESCLEAREDINDEX] * (double)rows_eroded[0] + 
                //weights[HEIGHTINDEX] * (double)aggregateHeight(f) + 
                //weights[BUMPINESSINDEX] * (double)bumpiness(f) +
                weights[HOLESINDEX] * (double)numOfHoles(f) + 
                weights[PILEHEIGHTINDEX] * (double)pileHeight(f) + 
                weights[CONHOLESINDEX] * (double)numOfConnectedHoles(f) +
                weights[ALTITUDEDIFINDEX] * (double)altitudeDifference(f) +
                weights[SUMOFWELLDEPTHSINDEX] * (double)wells[0] + 
                weights[MAXWELLDEPTHINDEX] * (double)wells[1] +
                weights[LANDINGHEIGHTINDEX] * (double)landingHeight(f) +
                weights[BLOCKCOUNTINDEX] * (double)blocks[0] +
                weights[WBLOCKCOUNTINDEX] * (double)blocks[1] +
                weights[ROWTRANSINDEX] * (double)rowTransitions(f) +
                weights[COLTRANSINDEX] * (double)colTransitions(f) + 
                weights[ERODEDBLOCKINDEX] * (double)rows_eroded[1];
        return util;
    }

    /** Used to calculate the utility for particle swarm optimization
     *   Updates the particle with some of the features in order to calculate it's fitness
     * @param f The updated field with the current move
     * @return A double array containing the utility [0], pileHeight[1], number of holes [2], row transitions [3], and column transitions [4]
     */
    public double[] calculateUtility_PSO(NewField f) {
        double[] util = new double[5];
        
        int[] wells = sumAndMaxOfWellDepths(f);
        int[] blocks = blockCount_Weighted(f);
        int[] rows_eroded = linesCleared(f);
        int pileHeight = pileHeight(f);
        int numOfHoles = numOfHoles(f);
        int rowTransitions = rowTransitions(f); 
        int colTransitions = colTransitions(f);
        
        util[1] = (double)pileHeight;
        util[2] = (double)numOfHoles;
        util[3] = (double)rowTransitions;
        util[4] = (double)colTransitions;
        
        util[0] =   weights[LINESCLEAREDINDEX] * (double)rows_eroded[0] + 
                    //weights[HEIGHTINDEX] * (double)aggregateHeight(f) + 
                    //weights[BUMPINESSINDEX] * (double)bumpiness(f) +
                    weights[HOLESINDEX] * (double)numOfHoles + 
                    weights[PILEHEIGHTINDEX] * (double)pileHeight + 
                    weights[CONHOLESINDEX] * (double)numOfConnectedHoles(f) +
                    weights[ALTITUDEDIFINDEX] * (double)altitudeDifference(f) +
                    weights[SUMOFWELLDEPTHSINDEX] * (double)wells[0] + 
                    weights[MAXWELLDEPTHINDEX] * (double)wells[1] +
                    weights[LANDINGHEIGHTINDEX] * (double)landingHeight(f) +
                    weights[BLOCKCOUNTINDEX] * (double)blocks[0] +
                    weights[WBLOCKCOUNTINDEX] * (double)blocks[1] +
                    weights[ROWTRANSINDEX] * (double)rowTransitions +
                    weights[COLTRANSINDEX] * (double)colTransitions + 
                    weights[ERODEDBLOCKINDEX] * (double)rows_eroded[1];
        return util;
    }
    
    /** Sums the height of each column
     * @author JoshMcManus
     * @param field The current Tetris field
     * @return sum The sum of each column's height
     */
    public int aggregateHeight(NewField f) {
        int sum = 0;
        // Loop through each column and find the first spot with a block
        for (int i = 0; i < f.top.length; i++) {
            sum += f.top[i];
        }
        return sum;
    }

    /** Calculates the number of rows cleared and eroded block count
     *     Eroded blocks are those that are placed but then removed by the cleared rows
     * @param f The updated field after placing a piece
     * @return Number of rows cleared and eroded block count
     */
    public static int[] linesCleared(NewField f) {
          //check for full rows - starting at the top
            int rowsCleared = 0;
            int curTurn = f.turn;
            Boolean cleared;
            int erodedBlocks;
            int erodedBlocksTotal = 0;
            for (int row = 0; row < State.ROWS; row++) {
                cleared = true;
                erodedBlocks = 0;
                for (int col = 0; col < State.COLS; col++) {
                    if (f.field[row][col] == 0) {
                        cleared = false;
                        break;
                    } else if (f.field[row][col] == curTurn) {
                        // If the block is part of the piece just placed, increment the number of cleared blocks
                        erodedBlocks++;
                    }
                }
                if (cleared) {
                    rowsCleared++;
                    erodedBlocksTotal += erodedBlocks;
                    for(int col = 0; col < State.COLS; col++) {
                        //slide down all bricks
                        for(int i = row; i < f.top[col]; i++) {
                            f.field[i][col] = f.field[i+1][col];
                        }
                        //lower the top
                        f.top[col]--;
                        while(f.top[col]>=1 && f.field[f.top[col]-1][col]==0)   f.top[col]--;
                    }
                }
            }
            int[] rows_eroded = {rowsCleared, erodedBlocksTotal * rowsCleared};
            return rows_eroded;
        }

    /** Calculates the bumpiness of a field
     * Bumpiness is defined as the height difference between adjacent columns
     * @param f The updated field after placing a piece
     * @return The sum of differences between each pair of adjacent columns
     */
    public static int bumpiness(NewField f) {
        int sum = 0;
        for (int i = 0; i < State.COLS - 1; i++) {
            sum += Math.abs(f.top[i] - f.top[i + 1]);
        }
        return sum;
    }

    /** Calculates the number of holes in the new field
     * A hole is defined as any unoccupied space that has some piece above it
     * @param f The updated field after placing a piece
     * @return Total number of holes in the updated field
     */
    public static int numOfHoles(NewField f) {
        int numHoles = 0;
        for (int col = 0; col < State.COLS; col++) {
            for (int row = f.top[col] - 1; row >= 0; row--) {
                if (f.field[row][col] == 0) {
                    numHoles++;
                }
            }
        }
        return numHoles;
    }

    /** Calculates the maximum height of the field
     * 
     * @param f The updated field after placing a piece
     * @return Max height of the field
     */
    public static int pileHeight(NewField f) {
        int max = 0;
        for (int i = 0; i < State.COLS; i++) {
            if (f.top[i] > max) {
                max = f.top[i];
            }
        }
        return max;
    }
    
    /** Calculates the number of connected holes in a field
     * A connected hole is any hole as defined above, but two vertically
     *     adjacent holes are considered as one
     * @param f The updated field after placing a piece
     * @return The total number of connected hoels in the field
     */
    public static int numOfConnectedHoles(NewField f) {
        int numHoles = 0;
        for (int col = 0; col < State.COLS; col++) {
            for (int row = f.top[col] - 1; row >= 0; row--) {
                if (f.field[row][col] == 0) {
                    // Continue down the column until another piece is found
                    while (row >= 0 && f.field[row][col] == 0) {
                        row--;
                    }
                    numHoles++;
                }
            }
        }
        return numHoles;
    }

    /** Calculates the difference between the highest and lowest point
     * 
     * @param f The updated field after placing a piece
     * @return The max "top" value minus the min value
     */
    public static int altitudeDifference(NewField f) {
        int i;  
        int min; int max;
        /* If array has even number of elements then 
          initialize the first two elements as minimum and 
          maximum */
        if (State.COLS % 2 == 0)
        {         
          if (f.top[0] > f.top[1])     
          {
            max = f.top[0];
            min = f.top[1];
          }  
          else
          {
            min = f.top[0];
            max = f.top[1];
          }
          i = 2;  // set the starting index for loop
        }  
       
         /* If array has odd number of elements then 
          initialize the first element as minimum and 
          maximum */
        else
        {
          min = f.top[0];
          max = f.top[0];
          i = 1;  //set the starting index for loop
        }
         
        /* In the while loop, pick elements in pair and 
           compare the pair with max and min so far */   
        while (i < State.COLS-1)  
        {          
          if (f.top[i] > f.top[i+1])          
          {
            if(f.top[i] > max)        
              max = f.top[i];
            if(f.top[i+1] < min)          
              min = f.top[i+1];        
          } 
          else        
          {
            if (f.top[i+1] > max)        
              max = f.top[i+1];
            if (f.top[i] < min)          
              min = f.top[i];        
          }        
          i += 2; /* Increment the index by 2 as two 
                     elements are processed in loop */
        }
        return max - min;
    }
    
    /** Calculate the sum of well depths and max well depth
     * Well depth is defined as any column where the two adjacent columns are higher than it,
     *     the depth is equal to the number of spaces where blocks sit on both sides.
     * @param f The updated field after placing a piece
     * @return An array of two values with the sum in position 0 and the max value in pos 1
     */
    public static int[] sumAndMaxOfWellDepths(NewField f) {
        int sum = 0;
        int max = 0;
        int[] top = f.top;
        
        // Check all inner columns first
        for (int column = 1; column < top.length - 1; ++column) {
            // If the top of the current column is lower than the two adjacent columns
            if (top[column - 1] > top[column] && top[column] < top[column + 1]) {
                int depth = Math.min(top[column-1] - top[column], top[column+1] - top[column]); 
                sum += depth;
                max = Math.max(max, depth);
            }
        }
        // Now check the left most column (we treat the wall as a max height
        if (top[0] < top[1]) {
            max = Math.max(max, top[1] - top[0]);
            sum += top[1] - top[0];
        }
        // Finally, check the rightmost column
        if (top[top.length - 1] < top[top.length - 2]) {
            max = Math.max(max, top[top.length - 2] - top[top.length - 1]);
            sum += (top[top.length - 2] - top[top.length - 1]);
        }

        int[] ret = {sum, max};
        return ret;
    }
    
    /** Calculates the landing height of the most recently placed piece
     * @param f The updated field after placing a piece
     * @return The highest row the placed piece occupies
     */
    public static int landingHeight(NewField f) {
        int max = 0;
        // Loop through each column looking for where the most recent piece was placed
        for (int i = 0; i < State.COLS; i++) {
            if (f.top[i] == f.turn) {
                // Find the highest point this landed piece occupies
                if (f.top[i] > max) {
                    max = f.top[i];
                }
            }
        }
        return max;
    }
    
    /** Calculates the total number of blocks in the field and the weighted sum
     *     The weight of a block is its row (ex. a block in row 1 has 1 weight, row 8 has a weight of 8)
     * @param f The updated field after placing a piece
     * @return Total number of occupied cells in the grid
     */
    public static int[] blockCount_Weighted(NewField f) {
        int count = 0;
        int weights = 0;
        for (int row = 0; row < State.ROWS; row++) {
            for (int col = 0; col < State.COLS; col++) {
                if (f.field[row][col] > 0) {
                    count++;
                    weights += (row + 1);
                }
            }
        }
        int[] ret = {count, weights};
        return ret;
    }
    
    /** Calculates the number of transitions in each row
     * A transition is the changing from a filled block to an empty one (and vice-versa)
     * @param f The updated field after placing a piece
     * @return Total number of row transitions in the field
     */
    public static int rowTransitions(NewField f) {
        int count = 0;
        for (int row = 0; row < State.ROWS; row++) {
            boolean isFilled = true; // Start filled since the edges are considered filled
            for (int col = 0; col < State.COLS; col++) {
                // If we detect a state change (filled->not filled or not filled->filled)
                if ((f.field[row][col] > 0) != isFilled) {
                    // Update the current status and increment the count
                    isFilled = !isFilled; 
                    count++;
                }
            }
            // If the last column is not filled, then there is a state change between the edge
            if (!isFilled) {
                count++;
            }
        }
        return count;
    }
    
    public static int colTransitions(NewField f) {
        int count = 0;
        for (int col = 0; col < State.COLS; col++) {
            boolean isFilled = true; // Start filled since the edges are considered filled
            for (int row = 0; row < State.ROWS; row++) {
                // If we detect a state change (filled->not filled or not filled->filled)
                if ((f.field[row][col] > 0) != isFilled) {
                    // Update the current status and increment the count
                    isFilled = !isFilled; 
                    count++;
                }
            }
            // If the last row is filled, then there is a state change between the top (which is empty)
            if (isFilled) {
                count++;
            }
        }
        return count;
    }
    
    /** Sets the weights to be used by the AI
     * @author Josh
     * @param weights Predetermined weights retrieved from an external file
     */
    public void setWeights(double[] weights) {
        
        // Need to have all weights filled in
        if (weights.length != NUMOFFEATURES) {
            this.weights = new double[NUMOFFEATURES];
        } else {
            this.weights = weights.clone();
        }
    }
}