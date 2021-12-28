package LeetCodeExcise;

import java.util.Arrays;

public class LeetCodeHardSolution {
    public static void main(String[] args) {
        int[][] data = new int[][]{{0, 5},{6,16}};
        int[] inter = new int[]{6,11};
        int[][] res = LeetCodeHardSolution.insert(data,inter);
        for(int i = 0;i<res.length;i++){
            for(int j = 0;j<res[0].length;j++){
                System.out.println(res[i][j]);
            }
        }

    }

    /**
     * 57. 插入区间
     * @param intervals
     * @param newInterval
     * @return
     * [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
     * [4,8]
     */
    public static int[][] insert(int[][] intervals, int[] newInterval) {
        int length = intervals.length;
        if(length==0){
            return new int[][]{newInterval};
        }
        int[][] mergeInterRes = mergeInterNums(intervals,newInterval);

        // 对获得的区间数组进行排序
        Arrays.sort(mergeInterRes, (o1, o2) -> {
            if (o1[0]==o2[0]) return o1[1]-o2[1];
            return o1[0]-o2[0];
        });

        int index = 0;
        int left = mergeInterRes[0][0],right = mergeInterRes[0][1];
        for(int i = 1;i < mergeInterRes.length;i++){
            if(left > mergeInterRes[i][1] || right < mergeInterRes[i][0]){
                mergeInterRes[index] = new int[]{left,right};
                index++;
                left = mergeInterRes[i][0];
                right = mergeInterRes[i][1];
            } else if(mergeInterRes[i][0] >= left && mergeInterRes[i][1] <= right){

            } else if(mergeInterRes[i][0] >= left && mergeInterRes[i][0] <= right && mergeInterRes[i][1] >= right){
                right = mergeInterRes[i][1];
            } else if(mergeInterRes[i][0] <= left && mergeInterRes[i][1] >= left && mergeInterRes[i][1] <= right){
                left = mergeInterRes[i][0];
            } else if(mergeInterRes[i][0] <= left && mergeInterRes[i][1] >= right){
                right = mergeInterRes[i][1];
                left = mergeInterRes[i][0];
            }
        }
        mergeInterRes[index] = new int[]{left,right};
        int[][] result = new int[index+1][2];
        for(int i =0;i<index+1;i++){
            result[i] = mergeInterRes[i];
        }
        return result;
    }
    public static int[][] mergeInterNums(int[][] intervals, int[] newInterval){
        int length = intervals.length;
        int[][] res = new int[length+1][2];
        res[length][0] = Integer.MIN_VALUE;
        res[length][1] = Integer.MIN_VALUE;
        for(int i = 0;i < length; i++) {
            if(newInterval[0] == Integer.MIN_VALUE && newInterval[1] == Integer.MIN_VALUE){
                break;
            }
            if(intervals[i][0] >= newInterval[0] && intervals[i][1] <= newInterval[1]){
                intervals[i][0] = newInterval[0];
                intervals[i][1] = newInterval[1];
                newInterval[0] = Integer.MIN_VALUE;
                newInterval[1] = Integer.MIN_VALUE;
            } else if(intervals[i][0] >= newInterval[0] && intervals[i][0] <= newInterval[1] && intervals[i][1] >= newInterval[1]){
                intervals[i][0] = newInterval[0];
                newInterval[0] = Integer.MIN_VALUE;
                newInterval[1] = Integer.MIN_VALUE;
            } else if(intervals[i][0] <= newInterval[0] && intervals[i][1] >= newInterval[0] && intervals[i][1] <= newInterval[1]){
                intervals[i][1] = newInterval[1];
                newInterval[0] = Integer.MIN_VALUE;
                newInterval[1] = Integer.MIN_VALUE;
            } else if(intervals[i][0] <= newInterval[0] && intervals[i][1] >= newInterval[1]){
                newInterval[0] = Integer.MIN_VALUE;
                newInterval[1] = Integer.MIN_VALUE;
            } else if(intervals[i][0] >= newInterval[1] || intervals[i][1] <= newInterval[0]){
                res[length][0] = newInterval[0];
                res[length][1] = newInterval[1];
                newInterval[0] = Integer.MIN_VALUE;
                newInterval[1] = Integer.MIN_VALUE;
            }
        }
        if(res[length][0] == Integer.MIN_VALUE && res[length][1] == Integer.MIN_VALUE){
            return intervals;
        }else{
            for(int i = 0; i < length;i++){
                res[i] = intervals[i];
            }
            return res;
        }
    }
}
