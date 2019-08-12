package Catarina;
import java.util.Random;
import java.util.ArrayList;

import core.game.Observation;
import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Utils;

/**
 *@version Version3 with Loss Avoidance
 */
public class SingleTreeNode
{
    private final double HUGE_NEGATIVE = -10000000.0;
    private final double HUGE_POSITIVE =  10000000.0;
    public double epsilon = 1e-6;
    public SingleTreeNode parent;
//    public ArrayList<ArrayList<SingleTreeNode> > children;// TODO: define class Move which includes an ArrayList<SingleTreeNode>, use Move[] to replace ArrayList<ArrayList<SingleTreeNode> > children;
    public Move[] moves;
    public double bestVal;
    public double worstVal;
    public int nVisits;
    public Random m_rnd;
    public int m_depth;
    protected double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    public int childIdx;

    public int num_actions;
    Types.ACTIONS[] actions;
    public int ROLLOUT_DEPTH = 10;
    public double K = Math.sqrt(2);

    public StateObservation rootState;
    public long StateID;
    public int bestActionIDX = 0;
    public SingleTreeNode(Random rnd, int num_actions, Types.ACTIONS[] actions, StateObservation state) {
        this(null, -1, rnd, num_actions, actions, state);
    }

    public SingleTreeNode(SingleTreeNode parent, int childIdx, Random rnd, int num_actions, Types.ACTIONS[] actions, StateObservation state) {
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        moves = new Move[num_actions];
        for(int i = 0; i < moves.length; i++){
            moves[i] = new Move();
        }
        bestVal = -Double.MAX_VALUE;
        worstVal = Double.MAX_VALUE;
        this.childIdx = childIdx;
        this.StateID = SingleTreeNode.calculateStateId(state);


        if(parent != null)
            m_depth = parent.m_depth+1;
        else
            m_depth = 0;
    }

    public void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = elapsedTimer.remainingTimeMillis();
        int numIters = 0;

        int remainingLimit = 5;
        while(remaining > 2*avgTimeTaken && remaining > remainingLimit){
        //while(numIters < Agent.MCTS_ITERATIONS){

            StateObservation state = rootState.copy();

            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy(state);
            double delta = selected.rollOut(state);
            backUp(selected, delta);

            numIters++;
            acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
            //System.out.println(elapsedTimerIteration.elapsedMillis() + " --> " + acumTimeTaken + " (" + remaining + ")");
            avgTimeTaken  = acumTimeTaken/numIters;
            remaining = elapsedTimer.remainingTimeMillis();
        }
//        System.out.println(numIters);
//        System.out.println(Agent.mctsPlayer.pastLocations.size());
    }

    public SingleTreeNode treePolicy(StateObservation state) {

        SingleTreeNode cur = this;

        while (!state.isGameOver() && cur.m_depth < ROLLOUT_DEPTH)
        {
            if (cur.notFullyExpanded()) {
                return cur.expand(state);

            } else {
                SingleTreeNode next = cur.uct(state);
                cur = next;
            }
        }

        return cur;
    }


    public SingleTreeNode expand(StateObservation state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < moves.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && moves[i].states == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        state.advance(actions[bestAction]);

        SingleTreeNode tn = new SingleTreeNode(this,bestAction,this.m_rnd,num_actions, actions, state);
        moves[bestAction].states.add(tn);
        return tn;
    }

    public SingleTreeNode uct(StateObservation state) {

        SingleTreeNode selected = null;
        int highestAct = -1, bestAct = -1;
        double highestVal = -Double.MAX_VALUE;
        for (int actionIndex = 0; actionIndex < moves.length; actionIndex++ ) {
            double actionVal = 0;
            int actionNbVisits = 0;
/*            for (SingleTreeNode child : moves[actionIndex].states ) {

                actionVal += child.bestVal * child.nVisits;
                actionNbVisits += child.nVisits;

            }*/
            actionVal = moves[actionIndex].avrValue;
            actionVal = Utils.normalise(actionVal, bounds[0], bounds[1]);
            double uctValue = actionVal +
                    K * Math.sqrt(Math.log(this.nVisits + 1) / (actionNbVisits + this.epsilon));

            uctValue = Utils.noise(uctValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > highestVal ) {
                highestAct = actionIndex;
                highestVal  = uctValue;
                if(moves[actionIndex].worstValue > HUGE_NEGATIVE){
                    bestAct = highestAct;
                }
            }
        }
        if (highestAct == -1)
        {
            throw new RuntimeException("Warning! returning null: " + highestVal  + " : " + this.moves.length + " " +
            + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        if(bestAct == -1){
            bestAct = highestAct;
        }
        state.advance(actions[bestAct]);
        selected = createChild(bestAct, state);

        return selected;
    }


    public double rollOut(StateObservation state)
    {
        int thisDepth = this.m_depth;

        while (!finishRollout(state,thisDepth)) {

            int action = m_rnd.nextInt(num_actions);
            state.advance(actions[action]);
            thisDepth++;
        }


        double delta = value(state);

        if(delta < bounds[0])
            bounds[0] = delta;
        if(delta > bounds[1])
            bounds[1] = delta;

        //double normDelta = utils.normalise(delta ,lastBounds[0], lastBounds[1]);

        return delta;
    }

    public double value(StateObservation a_gameState) {

        boolean gameOver = a_gameState.isGameOver();
        Types.WINNER win = a_gameState.getGameWinner();
        double rawScore = a_gameState.getGameScore() ;

        if(gameOver && win == Types.WINNER.PLAYER_LOSES)
            rawScore += HUGE_NEGATIVE;

        if(gameOver && win == Types.WINNER.PLAYER_WINS)
            rawScore += HUGE_POSITIVE;

        Long PosId = calculatePosId(a_gameState);
        if(Agent.mctsPlayer.pastLocations.containsKey(PosId)) rawScore -= (double)(10* Agent.mctsPlayer.pastLocations.get(PosId));
        return rawScore;
    }

    public boolean finishRollout(StateObservation rollerState, int depth)
    {
        if(depth >= ROLLOUT_DEPTH)      //rollout end condition.
            return true;

        if(rollerState.isGameOver())               //end of game
            return true;

        return false;
    }
/*

    public void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.bestVal += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }
*/
/**
 * backUp method using Loss Avoidance. Update both SingleTreeNode and Move values.
 */
    public void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        n.bestVal = result;
        n.nVisits ++;
        n.worstVal = result;
        while(n != null)
        {
            if(n.parent != null ){
                int childNum = n.childIdx;
                Move action = n.parent.moves[childNum];
                //update values in moves[childNum]
                action.avrValue =  ( action.avrValue * action.visitNum + result) / (action.visitNum + 1);
                action.visitNum ++;
                if(n.worstVal < action.worstValue){
                    action.worstValue = n.worstVal;
                }
                //update values in parent SingleTreeNode
                n.parent.nVisits++;
                if(action.avrValue > n.parent.bestVal) {
                    n.parent.bestVal = action.avrValue;
                    n.parent.worstVal = action.worstValue;
                }
            }
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }


    public int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i = 0; i < moves.length; i++) {

            if(moves[i].states != null)
            {
                double actionNVisit = 0;
                for(SingleTreeNode child : moves[i].states ) {
                    actionNVisit += child.nVisits;
                }
                if(first == -1) {
                    first = actionNVisit;
                }
                else if(first != actionNVisit)
                {
                    allEqual = false;
                }

                double actonVal = actionNVisit;
                actonVal = Utils.noise(actonVal, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (actonVal > bestValue) {
                    bestValue = actonVal;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }
        bestActionIDX = selected;
        return selected;
    }

    public int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<moves.length; i++) {

            if(moves[i].states != null) {
                //double tieBreaker = m_rnd.nextDouble() * epsilon;
                double actionVal = 0;
                double actionNvisit = 0;
                for(SingleTreeNode child: moves[i].states){
                    actionVal += child.bestVal;
                    actionNvisit += child.nVisits;
                }
                actionVal = actionVal / (actionNvisit+ this.epsilon);
                actionVal = Utils.noise(actionVal, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (actionVal > bestValue) {
                    bestValue = actionVal;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    public boolean notFullyExpanded() {
        for (Move action : moves) {
            if (action == null) {
                return true;
            }
        }

        return false;
    }

//method created by number27

    public static long calculateStateId(StateObservation stateObs) {
//        stateObs = stateObs.copy();
        long h = 1125899906842597L;
  /*      ArrayList<Observation>[][] observGrid = stateObs.getObservationGrid();

        for (int y = 0; y < observGrid[0].length; y++) {
            for (int x = 0; x < observGrid.length; x++) {
                for (int i = 0; i < observGrid[x][y].size(); i++) {
                    Observation observ = observGrid[x][y].get(i);

                    h = 31 * h + x;
                    h = 31 * h + y;
                    h = 31 * h + observ.category;
                    h = 31 * h + observ.itype;
                }
            }
        }
*/
        ArrayList<Observation>[] npcList = stateObs.getNPCPositions();
        if(npcList != null){
            for(int i = 0; i < npcList.length; i++){
                for(Observation obs : npcList[i]){
               h = 31 * h + obs.category;
               h = 31 * h + obs.itype;
               h = 31 * h + (int)(obs.position.x / stateObs.getBlockSize());
               h = 31 * h + (int)(obs.position.y / stateObs.getBlockSize());
                }
            }
        }

        ArrayList<Observation>[] movList = stateObs.getMovablePositions();
        if(movList != null){
            for(int i = 0; i < movList.length; i++){
                for(Observation obs : movList[i]){
                    h = 31 * h + obs.category;
                    h = 31 * h + obs.itype;
                    h = 31 * h + (int)(obs.position.x / stateObs.getBlockSize());
                    h = 31 * h + (int)(obs.position.y / stateObs.getBlockSize());
                }
            }
        }

        h = 31 * h + (int) (stateObs.getAvatarPosition().x / stateObs.getBlockSize());
        h = 31 * h + (int) (stateObs.getAvatarPosition().y / stateObs.getBlockSize());
        h = 31 * h + stateObs.getAvatarType();
        h = 31 * h + stateObs.getAvatarResources().size();
        h = 31 * h + (int) (stateObs.getGameScore() * 100);

        return h;

    }


    public static long calculatePosId(StateObservation stateObs){
        long h = 1125899906842597L;
        h = 31 * h + (int) (stateObs.getAvatarPosition().x / stateObs.getBlockSize());
        h = 31 * h + (int) (stateObs.getAvatarPosition().y / stateObs.getBlockSize());
        h = 31 * h + stateObs.getAvatarResources().size();


        return h;
    }

    /**
     * this.createChild() create a child node for this. If the state of child node already exist,
     * refer child to existing child, otherwise create a new one
     */
    public SingleTreeNode createChild(int actionIndex, StateObservation state ){
        SingleTreeNode child = null;
        long childStateID = SingleTreeNode.calculateStateId(state);
        for(SingleTreeNode existingChild : this.moves[actionIndex].states ){
            if(childStateID == existingChild.StateID) {
                child = existingChild;
                break;
            }
        }
        if(child == null){
            child = new SingleTreeNode(this, actionIndex,this.m_rnd,num_actions,actions,state );
            moves[actionIndex].states.add(child);
        }
        return child;
    }

    public void refresh(){
        for(Move action : this.moves){
            for(SingleTreeNode child : action.states){
                child.m_depth = this.m_depth +1;
            }
        }
    }


}

