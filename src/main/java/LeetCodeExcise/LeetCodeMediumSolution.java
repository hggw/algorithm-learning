package LeetCodeExcise;

import java.util.*;

// ## 表示难题，看了答案才懂；??表示难题，还没仔细看
public class LeetCodeMediumSolution {
    public static void main(String[] args) {

    }
    /**
     * 2. 两数相加
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        List<Integer> tmp1 = new ArrayList<>();
        while(l1 != null) {
            tmp1.add(l1.val);
            l1 = l1.next;
        }
        List<Integer> tmp2 = new ArrayList<>();
        while(l2 != null) {
            tmp2.add(l2.val);
            l2 = l2.next;
        }
        List<Integer> res = new ArrayList<>();
        int temp = 0;
        int i = 0;
        for(; i < Math.min(tmp1.size(),tmp2.size()); i++) {
            res.add((tmp1.get(i) + tmp2.get(i) + temp)%10);
            temp = (tmp1.get(i) + tmp2.get(i) + temp)/10;
        }
        if(tmp1.size() > tmp2.size()) {
            for(; i < tmp1.size(); i++ ) {
                res.add((tmp1.get(i) + temp)%10);
                temp = (tmp1.get(i) + temp)/10;
            }
        } else {
            for(; i < tmp2.size(); i++ ) {
                res.add((tmp2.get(i) + temp)%10);
                temp = (tmp2.get(i) + temp)/10;
            }
        }
        if(temp != 0) res.add(temp);
        ListNode ans = new ListNode();
        ListNode tmp = ans;
        for(int num : res) {
            tmp.next = new ListNode(num);
            tmp = tmp.next;
        }
        return ans.next;
    }

    /** ??
     * 127. 单词接龙
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    Map<String, Integer> wordId = new HashMap<>();
    List<List<Integer>> edge = new ArrayList<>();
    int nodeNum = 0;

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        for (String word : wordList) {
            addEdge(word);
        }
        addEdge(beginWord);
        if (!wordId.containsKey(endWord)) {
            return 0;
        }
        int[] dis = new int[nodeNum];
        Arrays.fill(dis, Integer.MAX_VALUE);
        int beginId = wordId.get(beginWord), endId = wordId.get(endWord);
        dis[beginId] = 0;

        Queue<Integer> que = new LinkedList<>();
        que.offer(beginId);
        while (!que.isEmpty()) {
            int x = que.poll();
            if (x == endId) {
                return dis[endId] / 2 + 1;
            }
            for (int it : edge.get(x)) {
                if (dis[it] == Integer.MAX_VALUE) {
                    dis[it] = dis[x] + 1;
                    que.offer(it);
                }
            }
        }
        return 0;
    }

    public void addEdge(String word) {
        addWord(word);
        int id1 = wordId.get(word);
        char[] array = word.toCharArray();
        int length = array.length;
        for (int i = 0; i < length; ++i) {
            char tmp = array[i];
            array[i] = '*';
            String newWord = new String(array);
            addWord(newWord);
            int id2 = wordId.get(newWord);
            edge.get(id1).add(id2);
            edge.get(id2).add(id1);
            array[i] = tmp;
        }
    }

    public void addWord(String word) {
        if (!wordId.containsKey(word)) {
            wordId.put(word, nodeNum++);
            edge.add(new ArrayList<>());
        }
    }

    /**
     * 129. 求根到叶子节点数字之和
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if(root==null){
            return 0;
        }
        return fun(root,0);
    }
    public int fun(TreeNode node,int value){
        value = value*10 + node.val;
        if(node.left==null && node.right==null){
            return value;
        }
        int sum = 0;
        if(node.left!=null){
            sum += fun(node.left,value);
        }
        if(node.right!=null){
            sum += fun(node.right,value);
        }
        return sum;
    }
}
