package Catarina;

import java.util.HashMap;
import java.lang.Long;
import java.lang.Integer;
import java.util.Random;

import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;


public class SingleMCTSPlayer
{


    /**
     * Root of the tree.
     */
    public SingleTreeNode m_root;

    /**
     * Random generator.
     */
    public Random m_rnd;

    public int num_actions;
    public Types.ACTIONS[] actions;

    public HashMap<Long, Integer> pastLocations;
    public SingleMCTSPlayer(Random a_rnd, int num_actions, Types.ACTIONS[] actions)
    {
        this.num_actions = num_actions;
        this.actions = actions;
        m_rnd = a_rnd;
        pastLocations = new HashMap<>();
    }

    /**
     * Inits the tree with the new observation state in the root.
     * @param a_gameState current state of the game.
     */
    public void init(StateObservation a_gameState)
    {
        //Set the game observation to a newly root node.
        boolean treeReused = false;
        long newStateID = SingleTreeNode.calculateStateId(a_gameState);
        Long PosID = SingleTreeNode.calculatePosId(a_gameState);
        if(pastLocations.containsKey(PosID) ) {
            pastLocations.replace(PosID, pastLocations.get(PosID)+1 );
            System.out.println(pastLocations.get(PosID));
        }
        else pastLocations.put(PosID, 1);
        if(m_root != null){
            for(SingleTreeNode child : m_root.moves[m_root.bestActionIDX].states){
                if(newStateID == child.StateID) {
                    m_root = child;
                    m_root.parent = null;
                    m_root.m_depth = 0;
                    m_root.refresh(); //reste the depth etc
                    treeReused = true;
//                    System.out.println("Tree reused");
                    break;
                }
            }
        }

        if(treeReused == false){
            m_root = new SingleTreeNode(m_rnd, num_actions, actions, a_gameState);
        }

        m_root.rootState = a_gameState;


    }

    /**
     * Runs MCTS to decide the action to take. It does not reset the tree.
     * @param elapsedTimer Timer when the action returned is due.
     * @return the action to execute in the game.
     */
    public int run(ElapsedCpuTimer elapsedTimer)
    {
        //Do the search within the available time.
        m_root.mctsSearch(elapsedTimer);

        //Determine the best action to take and return it.
        int action = m_root.mostVisitedAction();
        //int action = m_root.bestAction();
        return action;
    }

}
