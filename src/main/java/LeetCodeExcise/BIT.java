package LeetCodeExcise;

import java.util.Arrays;

public class BIT {
    int[] tree;
    int[] idRec;
    int n;

    public BIT(int n) {
        this.n = n;
        this.tree = new int[n];
        Arrays.fill(this.tree, Integer.MAX_VALUE);
        this.idRec = new int[n];
        Arrays.fill(this.idRec, -1);
    }

    public int lowbit(int k) {
        return k & (-k);
    }

    public void update(int pos, int val, int id) {
        while (pos > 0) {
            if (tree[pos] > val) {
                tree[pos] = val;
                idRec[pos] = id;
            }
            pos -= lowbit(pos);
        }
    }

    public int query(int pos) {
        int minval = Integer.MAX_VALUE;
        int j = -1;
        while (pos < n) {
            if (minval > tree[pos]) {
                minval = tree[pos];
                j = idRec[pos];
            }
            pos += lowbit(pos);
        }
        return j;
    }
}
