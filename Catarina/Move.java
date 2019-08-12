package Catarina;

import java.util.ArrayList;

public class Move {
    public ArrayList<SingleTreeNode> states;
    public double worstValue ;
    public double avrValue;
    public int visitNum ;

    public Move(){
        states = new ArrayList<SingleTreeNode>();
        worstValue = Double.MAX_VALUE;
        avrValue = 0;
        visitNum = 0;
    }
}
