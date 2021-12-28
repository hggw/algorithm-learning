package LeetCodeExcise;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TarjanSCC {
    List<Integer>[] edges;
    List<Integer>[] edgesId;
    int[] low;
    int[] dfn;
    List<Integer> ans;
    int n;
    int ts;

    public TarjanSCC(int n, List<Integer>[] edges, List<Integer>[] edgesId) {
        this.edges = edges;
        this.edgesId = edgesId;
        this.low = new int[n];
        Arrays.fill(low, -1);
        this.dfn = new int[n];
        Arrays.fill(dfn, -1);
        this.n = n;
        this.ts = -1;
        this.ans = new ArrayList<Integer>();
    }

    public List<Integer> getCuttingEdge() {
        for (int i = 0; i < n; ++i) {
            if (dfn[i] == -1) {
                getCuttingEdge(i, -1);
            }
        }
        return ans;
    }

    private void getCuttingEdge(int u, int parentEdgeId) {
        low[u] = dfn[u] = ++ts;
        for (int i = 0; i < edges[u].size(); ++i) {
            int v = edges[u].get(i);
            int id = edgesId[u].get(i);
            if (dfn[v] == -1) {
                getCuttingEdge(v, id);
                low[u] = Math.min(low[u], low[v]);
                if (low[v] > dfn[u]) {
                    ans.add(id);
                }
            } else if (id != parentEdgeId) {
                low[u] = Math.min(low[u], dfn[v]);
            }
        }
    }
}
