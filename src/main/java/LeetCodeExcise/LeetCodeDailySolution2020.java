package LeetCodeExcise;

import java.util.*;

public class LeetCodeDailySolution2020 {
    public static void main(String[] args) {
        int[][] num1 = new int[][]{{0,1,2,0},{3,4,5,2},{1,3,1,5}};
        setZeroes(num1);
    }

    /**
     * 82. 删除排序链表中的重复元素 II
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode res = new ListNode(0,head);
        ListNode ans = res;
        while (ans.next != null && ans.next.next != null) {
            if(ans.next.val == ans.next.next.val) {
                int tmp = ans.next.val;
                while(ans.next != null && ans.next.val == tmp) {
                    ans.next = ans.next.next;
                }
            } else {
                ans = ans.next;
            }
        }
        return res.next;
    }

    /**
     * 456. 132 模式
     * @param nums
     * @return
     */
    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        Deque<Integer> candidateK = new LinkedList<>();
        candidateK.push(nums[n - 1]);
        int maxK = Integer.MIN_VALUE;

        for (int i = n - 2; i >= 0; --i) {
            if (nums[i] < maxK) {
                return true;
            }
            while (!candidateK.isEmpty() && nums[i] > candidateK.peek()) {
                maxK = candidateK.pop();
            }
            if (nums[i] > maxK) {
                candidateK.push(nums[i]);
            }
        }

        return false;
    }

    /**
     * 73. 矩阵置零
     * @param matrix
     */
    public static void setZeroes(int[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        ArrayList<int[]> indexs = new ArrayList<>();
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(matrix[i][j] == 0) {
                    indexs.add(new int[]{i,j});
                }
            }
        }
        for(int[] index : indexs) {
            Arrays.fill(matrix[index[0]],0);
            for(int i = 0; i < row; i++) {
                matrix[i][index[1]] = 0;
            }
        }
    }

    /**
     * 92. 反转链表 II
     * @param head
     * @param left
     * @param right
     * @return
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if(left == right) return head;
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode pre = res;
        for(int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        ListNode cur = pre.next;
        ListNode next;
        for(int i = 0; i < right - left; i++) {
            next = cur.next;
            cur.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return res.next;
    }

    /**
     * 59. 螺旋矩阵 II
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        int[][] result = new int[n][n];
        int left = 0, right = n - 1, top = 0, bottom = n - 1;
        int count = 1;
        while(left <= right && top <= bottom) {
            for(int col = left; col <= right ; col++) {
                result[top][col] = count++;
            }
            for(int row = top + 1; row <= bottom; row++) {
                result[row][right] = count++;
            }
            for(int col = right - 1; col >= left; col--) {
                result[bottom][col] = count++;
            }
            for(int row = bottom - 1; row > top; row--) {
                result[row][left] = count++;
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return result;
    }

    /**
     * 54. 螺旋矩阵
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if(matrix.length <= 0 ) return result;
        int rows = matrix.length, cols = matrix[0].length;
        int left = 0, right = cols - 1, top = 0, bottom = rows - 1;
        while(left <= right && top <= bottom) {
            for(int col = left; col <= right; col++) {
                result.add(matrix[top][col]);
            }
            for(int row = top + 1; row <= bottom; row++) {
                result.add(matrix[row][right]);
            }
            if(left < right && top < bottom) {
                for(int col = right - 1; col > left; col--) {
                    result.add(matrix[bottom][col]);
                }
                for(int row = bottom; row > top; row--) {
                    result.add(matrix[row][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return result;
    }

    /**
     * 331. 验证二叉树的前序序列化
     * @param preorder
     * @return
     */
    public boolean isValidSerialization(String preorder) {
        int length = preorder.length();
        int index = 0, slots = 1;
        while(index < length) {
            if(slots == 0) {
                return false;
            }
            if(preorder.charAt(index) == ',') {
                index++;
            } else if(preorder.charAt(index) == '#') {
                slots--;
                index++;
            } else {
                while(index < length && preorder.charAt(index) != ',') {
                    index++;
                }
                slots++;
            }
        }
        return slots == 0;
    }

    /** ##
     * 224. 基本计算器
     * @param s
     * @return
     */
    public int calculate(String s) {
        Deque<Integer> ops = new LinkedList<>();
        ops.push(1);
        int sign = 1;

        int ret = 0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            if (s.charAt(i) == ' ') {
                i++;
            } else if (s.charAt(i) == '+') {
                sign = ops.peek();
                i++;
            } else if (s.charAt(i) == '-') {
                sign = -ops.peek();
                i++;
            } else if (s.charAt(i) == '(') {
                ops.push(sign);
                i++;
            } else if (s.charAt(i) == ')') {
                ops.pop();
                i++;
            } else {
                long num = 0;
                while (i < n && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + s.charAt(i) - '0';
                    i++;
                }
                ret += sign * num;
            }
        }
        return ret;
    }

    /**
     * 503. 下一个更大元素 II
     */
    public static int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ret = new int[n];
        Arrays.fill(ret, -1);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n * 2 - 1; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i % n]) {
                ret[stack.pop()] = nums[i % n];
            }
            stack.push(i % n);
        }
        return ret;
    }

    /**
     * 354. 俄罗斯套娃信封问题
     */
    public static int maxEnvelopes(int[][] envelopes) {
        int length = envelopes.length;
        if(length <= 0) return 0;
        Arrays.sort(envelopes,(o1,o2) -> {
            if(o1[0] != o2[0]) return o1[0] - o2[0];
            return o2[1] - o1[1];
        });
        List<Integer> f = new ArrayList<>();
        f.add(envelopes[0][1]);
        for (int i = 1; i < length; ++i) {
            int num = envelopes[i][1];
            if (num > f.get(f.size() - 1)) {
                f.add(num);
            } else {
                int index = binarySearch(f, num);
                f.set(index, num);
            }
        }
        return f.size();
    }

    public static int binarySearch(List<Integer> f, int target) {
        int low = 0, high = f.size() - 1;
        while (low < high) {
            int mid = (high - low) / 2 + low;
            if (f.get(mid) < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }


    /**
     * 896. 单调数列
     * @param A
     * @return
     */
    public static boolean isMonotonic(int[] A) {
        int length = A.length;
        int count1 = 0, count2 = 0;
        for(int i = 1; i < length; i++) {
            if(A[i] >= A[i - 1]) {
                count1++;
            }
            if(A[i] <= A[i - 1]) {
                count2++;
            }
        }
        return count1 == length - 1 || count2 == length - 1;
    }

    /**
     * 1052. 爱生气的书店老板
     */
    public static int maxSatisfied(int[] customers, int[] grumpy, int X) {
        int res = 0;
        int length = customers.length;
        for(int i = 0; i < length; i++) {
            if(grumpy[i] == 0) {
                res += customers[i];
            }
        }
        int increase = 0;
        for(int i = 0; i < X; i++) {
            increase += customers[i] * grumpy[i];
        }
        int maxIncrease = increase;
        for(int i = X ; i < length; i++) {
            increase = increase - customers[i -X] * grumpy[i -X] + customers[i] * grumpy[i];
            maxIncrease = Math.max(maxIncrease,increase);
        }
        return res + maxIncrease;
    }

    /**
     * 766. 托普利茨矩阵
     */
    public boolean isToeplitzMatrix(int[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                int tmp = matrix[i][j];
                int row = i + 1, col = j + 1;
                while (row < rows && col < cols) {
                    if (matrix[row][col] != tmp) {
                        return false;
                    } else {
                        row++;
                        col++;
                    }
                }
            }

        }
        return true;
    }

    /**
     * 1438. 绝对差不超过限制的最长连续子数组
     */
    public static int longestSubarray1(int[] nums, int limit) {
        TreeMap<Integer,Integer> map = new TreeMap<>();
        int length = nums.length;
        int left = 0, right = 0;
        int res = 0;
        while(right < length) {
            map.put(nums[right],map.getOrDefault(nums[right],0) + 1);
            while(map.lastKey() - map.firstKey() > limit) {
                map.put(nums[left],map.get(nums[left]) - 1);
                if(map.get(nums[left]) == 0) {
                    map.remove(nums[left]);
                }
                left++;
            }
            res = Math.max(res,right - left + 1);
            right++;
        }
        return res;
    }

    /**
     * 697. 数组的度
     */
    public int findShortestSubArray(int[] nums) {
        Map<Integer,int[]> res = new HashMap<>();
        int length = nums.length;
        for(int i = 0; i < length; i++) {
            if(res.containsKey(nums[i])) {
                res.get(nums[i])[0]++;
                res.get(nums[i])[2] = i;
            } else {
                res.put(nums[i],new int[]{1,i,i});
            }
        }
        int maxLen = 0, minLen = 0;
        for(Map.Entry<Integer,int[]> entry : res.entrySet()) {
            int[] arr = entry.getValue();
            if(maxLen < arr[0]) {
                maxLen = arr[0];
                minLen = arr[2] - arr[1] + 1;
            } else if(maxLen == arr[0]) {
                if (minLen > arr[2] - arr[1] + 1) {
                    minLen = arr[2] - arr[1] + 1;
                }
            }
        }
        return minLen;
    }

    /**
     * 995. K 连续位的最小翻转次数
     */
    public static int minKBitFlips(int[] A, int K) {
        int n = A.length;
        int[] diff = new int[n + 1];
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt += diff[i];
            if ((A[i] + revCnt) % 2 == 0) {
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                ++revCnt;
                --diff[i + K];
            }
        }
        return ans;
    }

    /**
     * 485. 最大连续1的个数
     */
    public int findMaxConsecutiveOnes(int[] nums) {
        int length = nums.length;
        int res = 0;
        int count = 0;
        for (int num : nums) {
            if (num == 1) {
                count++;
            } else {
                res = Math.max(res, count);
                count = 0;
            }
        }
        return res;
    }

    /**
     * 567. 字符串的排列
     */
    public boolean checkInclusion(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if(n > m) return false;
        int[] count = new int[26];
        for(char ch : s1.toCharArray()) {
            count[ch - 'a']--;
        }
        int left = 0;
        for(int right = 0; right < m; right++) {
            int x = s2.charAt(right) - 'a';
            count[x]++;
            while(count[x] > 0) {
                count[s2.charAt(left) - 'a']--;
                left++;
            }
            if(right - left + 1 == n) {
                return true;
            }
        }
        return false;
    }

    /**
     * 992. K 个不同整数的子数组
     */
    public static int subarraysWithKDistinct(int[] A, int K) {
        int length = A.length;
        int res = 0;
        for(int i = 0; i < length - K + 1; i++) {
            Map<Integer,Integer> count = new HashMap<>();
            for(int j = i; j < length; j++) {
                count.put(A[j],count.getOrDefault(A[j],0) + 1);
                if(count.size() == K) {
                    res++;
                } else if(count.size() > K) {
                    break;
                }
            }
        }
        return res;
    }
    public static int subarraysWithKDistinct1(int[] A, int K) {
        int n = A.length;
        int[] num1 = new int[n + 1];
        int[] num2 = new int[n + 1];
        int tot1 = 0, tot2 = 0;
        int left1 = 0, left2 = 0, right = 0;
        int res = 0;
        while (right < n) {
            if (num1[A[right]] == 0) {
                tot1++;
            }
            num1[A[right]]++;
            if (num2[A[right]] == 0) {
                tot2++;
            }
            num2[A[right]]++;
            while (tot1 > K) {
                num1[A[left1]]--;
                if (num1[A[left1]] == 0) {
                    tot1--;
                }
                left1++;
            }
            while (tot2 > K - 1) {
                num2[A[left2]]--;
                if (num2[A[left2]] == 0) {
                    tot2--;
                }
                left2++;
            }
            res += left2 - left1;
            right++;
        }
        return res;
    }


    /**
     * 978. 最长湍流子数组
     * 若 i <= k < j，当 k 为奇数时， A[k] > A[k+1]，且当 k 为偶数时，A[k] < A[k+1]；
     * 或 若 i <= k < j，当 k 为偶数时，A[k] > A[k+1] ，且当 k 为奇数时， A[k] < A[k+1]。
     */
    public static int maxTurbulenceSize(int[] arr) {
        int length = arr.length;
        int left = 0, right = 0;
        int maxLength = 0;
        while (right < length - 1) {
            if(left == right) {
                if(arr[left] == arr[left + 1]) {
                    left++;
                }
                right++;
            } else {
                if(arr[right - 1] > arr[right] && arr[right] < arr[right + 1]) {
                    right++;
                } else if(arr[right - 1] < arr[right] && arr[right] > arr[right + 1]) {
                    right++;
                } else {
                    left = right;
                }
            }
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }

    /**
     * 665. 非递减数列
     */
    public static boolean checkPossibility(int[] nums) {
        int length = nums.length;
        int count = 0;
        for(int i = 0; i < length - 1; i++) {
            int x = nums[i], y = nums[i + 1];
            if(x > y) {
                count++;
                if(count > 1) {
                    return false;
                }
                if(i > 0 && y < nums[i - 1]) {
                    nums[i + 1] = x;
                }
            }
        }
        return true;
    }

    /**
     * 1423. 可获得的最大点数
     */
    public int maxScore(int[] cardPoints, int k) {
        int length = cardPoints.length;
        int windowSize = length - k;
        int sum = 0;
        for(int i = 0; i < windowSize; i++) {
            sum += cardPoints[i];
        }
        int minSum = sum;
        for(int i = windowSize; i < length; i++) {
            sum += cardPoints[i] - cardPoints[i - windowSize];
            minSum = Math.min(minSum,sum);
        }
        return Arrays.stream(cardPoints).sum() - minSum;
    }
    public int equalSubstring(String s, String t, int maxCost) {
        int length = s.length();
        int[] cost = new int[length];
        for(int i = 0; i < length; i++) {
            cost[i] = Math.abs(s.charAt(i) - t.charAt(i));
        }
        int maxLength = 0;
        int start = 0, end = 0;
        int sum = 0;
        while (end < length) {
            sum += cost[end];
            while(sum > maxCost) {
                sum -= cost[start];
                start++;
            }
            maxLength = Math.max(maxLength,end - start + 1);
            end++;
        }
        return maxLength;
    }

    /**
     * 643. 子数组最大平均数 I
     */
    public double findMaxAverage(int[] nums, int k) {
        int length = nums.length;
        double maxk = 0;
        for(int i = 0; i < k; i++) {
            maxk += nums[i];
        }
        double res = maxk;
        for(int i = k; i < length; i++) {
            maxk = maxk - nums[i - k] + nums[i];
            res = Math.max(res,maxk);
        }

        return res/k;
    }

    /**
     * 480. 滑动窗口中位数
     */
    public static double[] medianSlidingWindow(int[] nums, int k) {
        int length = nums.length;
        double[] res = new double[length - k + 1];
        int[] tmp1 = new int[k];
        if(k % 2 == 0) {
            for(int i = 0; i < length - k + 1; i++) {
                tmp1 = Arrays.copyOfRange(nums,i,i+k);
                Arrays.sort(tmp1);
                res[i] = ((double) (tmp1[k/2 - 1]) + (double)(tmp1[k/2]))/2;
            }
        } else {
            for(int i = 0; i < length - k + 1; i++) {
                tmp1 = Arrays.copyOfRange(nums,i,i+k);
                Arrays.sort(tmp1);
                res[i] = tmp1[k/2];
            }
        }
        return res;
    }

    /**
     * 424. 替换后的最长重复字符
     */
    public int characterReplacement(String s, int k) {
        int[] num = new int[26];
        int length = s.length();
        int maxn = 0;
        int left = 0, right = 0;
        while(right < length) {
            num[s.charAt(right) - 'A']++;
            maxn = Math.max(maxn, num[s.charAt(right) - 'A']);
            if(right - left + 1 - maxn > k) {
                num[s.charAt(left) - 'A']--;
                left++;
            }
            right++;
        }
        return right - left;
    }

    /**
     * 888. 公平的糖果棒交换
     */
    public int[] fairCandySwap(int[] A, int[] B) {
        int sumA = Arrays.stream(A).sum();
        int sumB = Arrays.stream(B).sum();
        int delta = (sumA - sumB) / 2;
        Set<Integer> record = new HashSet<>();
        for(int num : A) {
            record.add(num);
        }
        int[] ans = new int[2];
        for(int y : B) {
            int x = y + delta;
            if(record.contains(x)) {
                ans[0] = x;
                ans[1] = y;
                break;
            }
        }
        return ans;
    }

    /**
     * 724. 寻找数组的中心索引
     */
    public int pivotIndex(int[] nums) {
        int length = nums.length;
        int sum = 0;
        int before = 0, after = 0;
        int index = 0;
        for (int num : nums) {
            sum += num;
        }
        while(index < length) {
            if(index == 0) {
                before = 0;
                after = sum - nums[index];
            } else {
                before += nums[index - 1];
                after = sum - before - nums[index];
            }
            if(before == after) {
                return index;
            }
            index++;
        }
        return -1;
    }

    /**
     * 989. 数组形式的整数加法
     */
    public static List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> res = new ArrayList<>();
        int length = A.length;
        for(int i = length - 1; i >= 0; i--) {
            int sum = A[i] + K%10;
            K /= 10;
            if(sum >= 10) {
                K++;
                sum -= 10;
            }
            res.add(sum);
        }
        while(K!=0) {
            res.add(K%10);
            K /= 10;
        }
        Collections.reverse(res);
        return res;
    }

    /** 难度逆天了，题目看懂都费劲
     * 1489. 找到最小生成树里的关键边和伪关键边
     */
    public List<List<Integer>> findCriticalAndPseudoCriticalEdges(int n, int[][] edges) {
        int m = edges.length;
        int[][] newEdges = new int[m][4];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < 3; ++j) {
                newEdges[i][j] = edges[i][j];
            }
            newEdges[i][3] = i;
        }
        Arrays.sort(newEdges, new Comparator<int[]>() {
            public int compare(int[] u, int[] v) {
                return u[2] - v[2];
            }
        });

        UnionFind1 uf = new UnionFind1(n);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        for (int i = 0; i < 2; ++i) {
            ans.add(new ArrayList<Integer>());
        }
        int[] label = new int[m];
        for (int i = 0; i < m;) {
            // 找出所有权值为 w 的边，下标范围为 [i, j)
            int w = newEdges[i][2];
            int j = i;
            while (j < m && newEdges[j][2] == newEdges[i][2]) {
                ++j;
            }

            // 存储每个连通分量在图 G 中的编号
            Map<Integer, Integer> compToId = new HashMap<Integer, Integer>();
            // 图 G 的节点数
            int gn = 0;

            for (int k = i; k < j; ++k) {
                int x = uf.findset(newEdges[k][0]);
                int y = uf.findset(newEdges[k][1]);
                if (x != y) {
                    if (!compToId.containsKey(x)) {
                        compToId.put(x, gn++);
                    }
                    if (!compToId.containsKey(y)) {
                        compToId.put(y, gn++);
                    }
                } else {
                    // 将自环边标记为 -1
                    label[newEdges[k][3]] = -1;
                }
            }

            // 图 G 的边
            List<Integer>[] gm = new List[gn];
            List<Integer>[] gmid = new List[gn];
            for (int k = 0; k < gn; ++k) {
                gm[k] = new ArrayList<Integer>();
                gmid[k] = new ArrayList<Integer>();
            }

            for (int k = i; k < j; ++k) {
                int x = uf.findset(newEdges[k][0]);
                int y = uf.findset(newEdges[k][1]);
                if (x != y) {
                    int idx = compToId.get(x), idy = compToId.get(y);
                    gm[idx].add(idy);
                    gmid[idx].add(newEdges[k][3]);
                    gm[idy].add(idx);
                    gmid[idy].add(newEdges[k][3]);
                }
            }

            List<Integer> bridges = new TarjanSCC(gn, gm, gmid).getCuttingEdge();
            // 将桥边（关键边）标记为 1
            for (int id : bridges) {
                ans.get(0).add(id);
                label[id] = 1;
            }

            for (int k = i; k < j; ++k) {
                uf.unite(newEdges[k][0], newEdges[k][1]);
            }

            i = j;
        }

        // 未标记的边即为非桥边（伪关键边）
        for (int i = 0; i < m; ++i) {
            if (label[i] == 0) {
                ans.get(1).add(i);
            }
        }

        return ans;
    }

    /**
     * 628. 三个数的最大乘积
     */
    public static int maximumProduct(int[] nums) {
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        for(int x : nums) {
            if(x < min1) {
                min2 = min1;
                min1 = x;
            } else if(x < min2) {
                min2 = x;
            }
            if(x > max1) {
                max3 = max2;
                max2 = max1;
                max1 = x;
            } else if(x > max2) {
                max3 = max2;
                max2 = x;
            } else if(x > max3) {
                max3 = x;
            }
        }
        return Math.max(min1 * min2 * max1, max1 * max2 * max3);
    }

    /**
     * 721. 账户合并
     */
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, Integer> emailToIndex = new HashMap<String, Integer>();
        Map<String, String> emailToName = new HashMap<String, String>();
        int emailsCount = 0;
        for (List<String> account : accounts) {
            String name = account.get(0);
            int size = account.size();
            for (int i = 1; i < size; i++) {
                String email = account.get(i);
                if (!emailToIndex.containsKey(email)) {
                    emailToIndex.put(email, emailsCount++);
                    emailToName.put(email, name);
                }
            }
        }
        UnionFind uf = new UnionFind(emailsCount);
        for (List<String> account : accounts) {
            String firstEmail = account.get(1);
            int firstIndex = emailToIndex.get(firstEmail);
            int size = account.size();
            for (int i = 2; i < size; i++) {
                String nextEmail = account.get(i);
                int nextIndex = emailToIndex.get(nextEmail);
                uf.union(firstIndex, nextIndex);
            }
        }
        Map<Integer, List<String>> indexToEmails = new HashMap<Integer, List<String>>();
        for (String email : emailToIndex.keySet()) {
            int index = uf.find(emailToIndex.get(email));
            List<String> account = indexToEmails.getOrDefault(index, new ArrayList<String>());
            account.add(email);
            indexToEmails.put(index, account);
        }
        List<List<String>> merged = new ArrayList<List<String>>();
        for (List<String> emails : indexToEmails.values()) {
            Collections.sort(emails);
            String name = emailToName.get(emails.get(0));
            List<String> account = new ArrayList<String>();
            account.add(name);
            account.addAll(emails);
            merged.add(account);
        }
        return merged;
    }

    /**
     * 1232.缀点成线
     */
    public boolean checkStraightLine(int[][] coordinates) {
        if(coordinates.length < 2) return false;
        Arrays.sort(coordinates, (o1,o2) -> {
            if(o1[0] != o2[0]) return o1[0] - o2[0];
            return o1[1] - o2[1];
        });
        int key = 0;
        if(coordinates[1][0] != coordinates[0][0]) {
            key = (coordinates[1][1] - coordinates[0][1]) / (coordinates[1][0] - coordinates[0][0]);
        } else {
            key = coordinates[0][0];
        }
        for(int i = 2; i < coordinates.length; i++) {
            if (coordinates[i][0] != coordinates[i - 1][0]) {
                if((coordinates[i][1] - coordinates[i - 1][1]) / (coordinates[i][0] - coordinates[i - 1][0]) != key) return false;
            } else {
                if(coordinates[i][0] != key) return false;
            }
        }
        return true;
    }

    /**
     * 947. 移除最多的同行或同列石头
     */
    public int removeStones(int[][] stones) {
        int n = stones.length;
        DisjointSetUnion1 dsu = new DisjointSetUnion1();
        for (int i = 0; i < n; i++) {
            dsu.unionSet(stones[i][0], stones[i][1] + 10000);
        }

        return n - dsu.numberOfConnectedComponent();
    }
    /**
     * 1018. 可被 5 整除的二进制前缀
     * [false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,false,false,true,true,true,true,false]
     */
    public static List<Boolean> prefixesDivBy5(int[] A) {
        List<Boolean> res = new ArrayList<>();
        int tmp = 0;
        for (int value : A) {
            tmp = ((tmp << 1) + value) % 5;
            res.add(tmp == 0);
        }
        return res;
    }

    /** 并查集我还不会呀，图论难受呀
     * 684. 冗余连接
     */
    public int[] findRedundantConnection(int[][] edges) {
        int nodesCount = edges.length;
        int[] parent = new int[nodesCount + 1];
        for (int i = 1; i <= nodesCount; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < nodesCount; i++) {
            int[] edge = edges[i];
            int node1 = edge[0], node2 = edge[1];
            if (find(parent, node1) != find(parent, node2)) {
                union(parent, node1, node2);
            } else {
                return edge;
            }
        }
        return new int[0];
    }
    public void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    public int find(int[] parent, int index) {
        if (parent[index] != index) {
            parent[index] = find(parent, parent[index]);
        }
        return parent[index];
    }

    /**
     * 1202. 交换字符串中的元素
     */
    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        DisjointSetUnion dsu = new DisjointSetUnion(s.length());
        for (List<Integer> pair : pairs) {
            dsu.unionSet(pair.get(0), pair.get(1));
        }
        Map<Integer, List<Character>> map = new HashMap<Integer, List<Character>>();
        for (int i = 0; i < s.length(); i++) {
            int parent = dsu.find(i);
            if (!map.containsKey(parent)) {
                map.put(parent, new ArrayList<Character>());
            }
            map.get(parent).add(s.charAt(i));
        }
        for (Map.Entry<Integer, List<Character>> entry : map.entrySet()) {
            Collections.sort(entry.getValue(), new Comparator<Character>() {
                public int compare(Character c1, Character c2) {
                    return c2 - c1;
                }
            });
        }
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < s.length(); i++) {
            int x = dsu.find(i);
            List<Character> list = map.get(x);
            sb.append(list.remove(list.size() - 1));
        }
        return sb.toString();
    }

    /**
     * 189. 旋转数组
     * [1,2,3,4,5,6,7] 和 k = 3
     * 输出: [5,6,7,1,2,3,4]
     */
    public static void rotate(int[] nums, int k) {
        int length = nums.length;
        if(k >= length) k = k % length;
        int[] tmp = new int[length];
        if (k >= 0) System.arraycopy(nums, length - k + 0, tmp, 0, k);
        for(int i = k; i < length; i++) {
            tmp[i] = nums[i - k];
        }
        System.arraycopy(tmp, 0, nums, 0, length);
    }

    /** ##应该会的
     * 547. 省份数量
     */
    public int findCircleNum(int[][] isConnected) {
        int provinces = isConnected.length;
        boolean[] visited = new boolean[provinces];
        int res = 0;
        for(int i = 0; i < provinces; i++) {
            if(!visited[i]) {
                dfs(isConnected,visited,provinces,i);
                res++;
            }
        }
        return res;
    }
    public void dfs(int[][] isConnected,boolean[] visited,int provinces,int i) {
        for(int j = 0; j < provinces; j++) {
            if(isConnected[i][j] == 1 && !visited[j]) {
                visited[j] = true;
                dfs(isConnected,visited,provinces,j);
            }
        }
    }

    /** ##这题涉及图结构，暂时不会
     * 399. 除法求值
     */
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int nvars = 0;
        Map<String, Integer> variables = new HashMap<>();

        int n = equations.size();
        for (int i = 0; i < n; i++) {
            if (!variables.containsKey(equations.get(i).get(0))) {
                variables.put(equations.get(i).get(0), nvars++);
            }
            if (!variables.containsKey(equations.get(i).get(1))) {
                variables.put(equations.get(i).get(1), nvars++);
            }
        }
        double[][] graph = new double[nvars][nvars];
        for (int i = 0; i < nvars; i++) {
            Arrays.fill(graph[i], -1.0);
        }
        for (int i = 0; i < n; i++) {
            int va = variables.get(equations.get(i).get(0)), vb = variables.get(equations.get(i).get(1));
            graph[va][vb] = values[i];
            graph[vb][va] = 1.0 / values[i];
        }

        for (int k = 0; k < nvars; k++) {
            for (int i = 0; i < nvars; i++) {
                for (int j = 0; j < nvars; j++) {
                    if (graph[i][k] > 0 && graph[k][j] > 0) {
                        graph[i][j] = graph[i][k] * graph[k][j];
                    }
                }
            }
        }

        int queriesCount = queries.size();
        double[] ret = new double[queriesCount];
        for (int i = 0; i < queriesCount; i++) {
            List<String> query = queries.get(i);
            double result = -1.0;
            if (variables.containsKey(query.get(0)) && variables.containsKey(query.get(1))) {
                int ia = variables.get(query.get(0)), ib = variables.get(query.get(1));
                if (graph[ia][ib] > 0) {
                    result = graph[ia][ib];
                }
            }
            ret[i] = result;
        }
        return ret;
    }

    /**
     * 830. 较大分组的位置
     */
    public static List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> res = new ArrayList<>();
        if(s.length() < 3) return res;
        int left = 0, right = 0;
        while(right < s.length()) {
            if(s.charAt(left) == s.charAt(right)) {
                right++;
            } else if (right - left >= 3) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(left);
                tmp.add(right - 1);
                res.add(tmp);
                left = right;
            } else {
                left = right;
            }
        }
        if(right - left >= 3) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(left);
            tmp.add(right - 1);
            res.add(tmp);
        }
        return res;
    }

    /**
     * 435. 无重叠区间
     * @param intervals
     * @return
     */
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        Arrays.sort(intervals, (o1,o2) -> {
            return o1[1] - o2[1];
        });

        int n = intervals.length;
        int right = intervals[0][1];
        int ans = 1;
        for (int i = 1; i < n; ++i) {
            if (intervals[i][0] >= right) {
                ++ans;
                right = intervals[i][1];
            }
        }
        return n - ans;
    }

    /**
     * 1046. 最后一块石头的重量
     */
    public static int lastStoneWeight(int[] stones) {
        if(stones.length == 1) return stones[0];
        Arrays.sort(stones);
        int temp = stones.length - 1,tmp = temp - 1;
        while(stones[tmp] != 0) {
            stones[temp] -= stones[tmp];
            stones[temp - 1] = 0;
            Arrays.sort(stones);
        }
        return stones[temp];
    }

    /** ##着实不会
     * 330. 按要求补齐数组
     */
    public int minPatches(int[] nums, int n) {
        int patches = 0;
        long x = 1;
        int length = nums.length, index = 0;
        while (x <= n) {
            if (index < length && nums[index] <= x) {
                x += nums[index];
                index++;
            } else {
                x *= 2;
                patches++;
            }
        }
        return patches;
    }

    /** ##
     * 188. 买卖股票的最佳时机 IV
     */
    public int maxProfit(int k, int[] prices) {
        if (prices.length == 0) return 0;
        int n = prices.length;
        k = Math.min(k, n / 2);
        int[] buy = new int[k + 1];
        int[] sell = new int[k + 1];

        buy[0] = -prices[0];
        sell[0] = 0;
        for (int i = 1; i <= k; ++i) {
            buy[i] = sell[i] = Integer.MIN_VALUE / 2;
        }

        for (int i = 1; i < n; ++i) {
            buy[0] = Math.max(buy[0], sell[0] - prices[i]);
            for (int j = 1; j <= k; ++j) {
                buy[j] = Math.max(buy[j], sell[j] - prices[i]);
                sell[j] = Math.max(sell[j], buy[j - 1] + prices[i]);
            }
        }

        return Arrays.stream(sell).max().getAsInt();
    }

    /** ##
     * 205. 同构字符串
     */
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> s2t = new HashMap<Character, Character>();
        Map<Character, Character> t2s = new HashMap<Character, Character>();
        int len = s.length();
        for (int i = 0; i < len; ++i) {
            char x = s.charAt(i), y = t.charAt(i);
            if ((s2t.containsKey(x) && s2t.get(x) != y) || (t2s.containsKey(y) && t2s.get(y) != x)) {
                return false;
            }
            s2t.put(x, y);
            t2s.put(y, x);
        }
        return true;
    }
    /**
     * 455. 分发饼干
     */
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int index1 = 0,index2 = 0,count = 0;
        while(index1 < g.length && index2 < s.length) {
            if(g[index1] <= s[index2]) {
                count++;
                index1++;
            }
            index2++;
        }
        return count;
    }

    /**
     * 135. 分发糖果
     */
    public int candy(int[] ratings) {
        int length = ratings.length;
        int[] res = new int[length];
        for(int i = 0; i < length; i++) {
            if(i > 0 && ratings[i] > ratings[i - 1]) {
                res[i] = res[i - 1] + 1;
            } else {
                res[i] = 1;
            }
        }
        int right = 0, ans = 0;
        for(int i = length - 1; i >= 0; i--) {
            if(i < length - 1 && ratings[i] > ratings[i + 1]) {
                right++;
            } else {
                right = 1;
            }
            ans += Math.max(res[i],right);
        }
        return ans;
    }

    /**
     * 387. 字符串中的第一个唯一字符
     */
    public int firstUniqChar(String s) {
        int[] data = new int[26];
        for(char ch : s.toCharArray()) {
            data[ch - 'a']++;
        }
        for(char ch : s.toCharArray()) {
            if(data[ch - 'a'] == 1) {
                return s.indexOf(ch);
            }
        }
        return -1;
    }
    public int firstUniqChar1(String s) {
        Map<Character,Integer> data = new HashMap<>();
        for(char ch : s.toCharArray()) {
            data.put(ch,data.getOrDefault(ch,0) + 1);
        }

        for(char ch : s.toCharArray()) {
            if(data.get(ch) == 1) {
                return s.indexOf(ch);
            }
        }
        return -1;
    }

    /** ##
     * 103. 二叉树的锯齿形层序遍历
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<List<Integer>>();
        if (root == null) return ans;
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        nodeQueue.offer(root);
        boolean isOrderLeft = true;
        while (!nodeQueue.isEmpty()) {
            Deque<Integer> levelList = new LinkedList<Integer>();
            int size = nodeQueue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode curNode = nodeQueue.poll();
                if (isOrderLeft) {
                    levelList.offerLast(curNode.val);
                } else {
                    levelList.offerFirst(curNode.val);
                }
                if (curNode.left != null) {
                    nodeQueue.offer(curNode.left);
                }
                if (curNode.right != null) {
                    nodeQueue.offer(curNode.right);
                }
            }
            ans.add(new LinkedList<Integer>(levelList));
            isOrderLeft = !isOrderLeft;
        }

        return ans;
    }

    /**
     * 746. 使用最小花费爬楼梯
     */
    public int minCostClimbingStairs(int[] cost) {
        if(cost == null) return 0;
        if(cost.length ==1) return cost[0];
        if(cost.length == 2) return Math.min(cost[0],cost[1]);
        int[] res = new int[cost.length];
        res[0] = cost[0];
        res[1] = cost[1];
        for(int i = 2; i < cost.length; i++) {
            res[i] = Math.min(res[i - 1], res[i - 2]) + cost[i];
        }
        return Math.min(res[cost.length-1], res[cost.length - 2]);
    }

    /**
     * 316. 去除重复字母
     */
    public String removeDuplicateLetters(String s) {
        boolean[] vis = new boolean[26];
        int[] num = new int[26];
        for (int i = 0; i < s.length(); i++) {
            num[s.charAt(i) - 'a']++;
        }
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!vis[ch - 'a']) {
                while (sb.length() > 0 && sb.charAt(sb.length() - 1) > ch) {
                    if (num[sb.charAt(sb.length() - 1) - 'a'] > 0) {
                        vis[sb.charAt(sb.length() - 1) - 'a'] = false;
                        sb.deleteCharAt(sb.length() - 1);
                    } else {
                        break;
                    }
                }
                vis[ch - 'a'] = true;
                sb.append(ch);
            }
            num[ch - 'a'] -= 1;
        }
        return sb.toString();
    }

    /**
     * 389. 找不同
     */
    public char findTheDifference(String s, String t) {
        int[] res = new int[26];
        for(char ch : s.toCharArray()) {
            res[ch - 'a']++;
        }
        for(char ch : t.toCharArray()) {
            res[ch - 'a']--;
            if(res[ch - 'a'] < 0) return ch;
        }
        return 'a';
    }

    /**
     * 714. 买卖股票的最佳时机含手续费
     */
    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }
    public int maxProfit1(int[] prices, int fee) {
        int n = prices.length;
        int buy = prices[0] + fee;
        int profit = 0;
        for (int i = 1; i < n; ++i) {
            if (prices[i] + fee < buy) {
                buy = prices[i] + fee;
            } else if (prices[i] > buy) {
                profit += prices[i] - buy;
                buy = prices[i];
            }
        }
        return profit;
    }

    /**
     * 290. 单词规律
     */
    public static boolean wordPattern(String pattern, String s) {
        String[] strs = s.split(" ");
        if(strs.length != pattern.length()) return false;
        Map<Character,String> res = new HashMap<>();
        for(int i = 0; i < pattern.length(); i++) {
            if(!res.containsKey(pattern.charAt(i)) && !res.containsValue(strs[i])) {
                res.put(pattern.charAt(i),strs[i]);
            } else if(!res.containsKey(pattern.charAt(i)) && res.containsValue(strs[i])) {
                return false;
            } else {
                if(!res.get(pattern.charAt(i)).equals(strs[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 738. 单调递增的数字
     */
    public static int monotoneIncreasingDigits(int N) {
        char[] strs = Integer.toString(N).toCharArray();
        int i = 1;
        while(i < strs.length && strs[i - 1] <= strs[i]) {
            i++;
        }
        if(i < strs.length) {
            while(i > 0 && strs[i - 1] > strs[i]) {
                strs[i - 1]--;
                i--;
            }
            i++;
            for(; i < strs.length; i++) {
                strs[i] = '9';
            }
        }
        return Integer.parseInt(new String(strs));
    }

    /**
     * 49. 字母异位词分组
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> res = new HashMap<>();
        for(String str : strs) {
            char[] tmp = str.toCharArray();
            Arrays.sort(tmp);
            String key = new String(tmp);
            List<String> list = res.getOrDefault(key, new ArrayList<>());
            list.add(str);
            res.put(key,list);
        }
        return new ArrayList<>(res.values());
    }
    /**
     * 649. Dota2 参议院
     */
     public static String predictPartyVictory(String senate) {
         int n = senate.length();
         Queue<Integer> radiant = new LinkedList<Integer>();
         Queue<Integer> dire = new LinkedList<Integer>();
         for (int i = 0; i < n; ++i) {
             if (senate.charAt(i) == 'R') {
                 radiant.offer(i);
             } else {
                 dire.offer(i);
             }
         }
         while (!radiant.isEmpty() && !dire.isEmpty()) {
             int radiantIndex = radiant.poll(), direIndex = dire.poll();
             if (radiantIndex < direIndex) {
                radiant.offer(radiantIndex + n);
             } else {
                dire.offer(direIndex + n);
             }
         }
         return !radiant.isEmpty() ? "Radiant" : "Dire";
     }
    //过了72/81
    public static String predictPartyVictory1(String senate) {
        int length = senate.length();
        if(length == 0) return null;
        if(length == 1) return senate.equals("R") ? "Radiant" : "Dire";
        char[] rdIndex = senate.toCharArray();
        int[] valid = new int[length];
        Arrays.fill(valid,1);
        int rCount = 0, dCount = 0;
        for(int i = 0; i < length; i++) {
            if(valid[i] == 1) {
                if(rdIndex[i] == 'R') rCount++;
                else dCount++;
            }
        }
        while(rCount * dCount != 0) {
            for(int i = 0; i < length; i++) {
                if(valid[i] == 1) {
                    for(int j = i + 1; j < length; j++) {
                        if(rdIndex[j] != rdIndex[i] && valid[j] == 1) {
                            valid[j] = 0;
                            break;
                        }
                    }
                }
            }
            rCount = 0;
            dCount = 0;
            for(int i = 0; i < length; i++) {
                if(valid[i] == 1) {
                    if(rdIndex[i] == 'R') rCount++;
                    else dCount++;
                }
            }
        }

        return rCount > dCount ? "Radiant" : "Dire";
    }

    /**
     * 860. 柠檬水找零
     */
    public boolean lemonadeChange(int[] bills) {
        int countFive = 0, countTen = 0;
        for(int num : bills) {
            if(num == 10) {
                countTen++;
                if(countFive <= 0) return false;
                else countFive--;
            }
            if(num == 20) {
                if((countFive*5 + countTen*10) >= 15 && countFive > 0) {
                    countFive--;
                    countTen--;
                } else {
                    return false;
                }
            }
            if(num == 5) countFive++;
        }
        return true;
    }

    /**
     * 62. 不同路径
     */
    // 采用递归，时间会超时
    public static int uniquePaths1(int m, int n) {
        if(m*n <= 0) return 0;
        if(m*n <= 2) return 1;
        int res = 0;
        return res + uniquePaths(m - 1,n) + uniquePaths(m,n - 1);
    }
    public static int uniquePaths(int m, int n) {
        if(m*n <= 2) return 1;
        int[][] data = new int[n][m];
        for(int i = 1; i < n; i++) {
            data[i][0] = 1;
        }
        for(int j = 1; j < m; j++) {
            data[0][j] = 1;
        }
        for(int i = 1; i < n; i++) {
            for(int j = 1; j < m; j++) {
                data[i][j] = data[i-1][j] + data[i][j-1];
            }
        }
        return data[n-1][m-1];
    }

    /** ##
     * 842. 将数组拆分成斐波那契序列
     */
    public List<Integer> splitIntoFibonacci(String S) {
        List<Integer> list = new ArrayList<Integer>();
        backtrack(list, S, S.length(), 0, 0, 0);
        return list;
    }
    public boolean backtrack(List<Integer> list, String S, int length, int index, int sum, int prev) {
        if (index == length) {
            return list.size() >= 3;
        }
        long currLong = 0;
        for (int i = index; i < length; i++) {
            if (i > index && S.charAt(index) == '0') {
                break;
            }
            currLong = currLong * 10 + S.charAt(i) - '0';
            if (currLong > Integer.MAX_VALUE) {
                break;
            }
            int curr = (int) currLong;
            if (list.size() >= 2) {
                if (curr < sum) {
                    continue;
                } else if (curr > sum) {
                    break;
                }
            }
            list.add(curr);
            if (backtrack(list, S, length, i + 1, prev + curr, curr)) {
                return true;
            } else {
                list.remove(list.size() - 1);
            }
        }
        return false;
    }

    /** 位操作秀一脸
     * 861. 翻转矩阵后的得分
     */
    public static int matrixScore(int[][] A) {
        int n = A.length, m = A[0].length;
        int res = n * (1 << (m -1));
        for(int j = 1; j < m; j++) {
            int count = 0;
            for (int i = 0; i < n; i++) {
                count += A[i][0] == 1 ? A[i][j] : 1 - A[i][j];
            }
            res += Math.max(count,n - count) * (1 << (m - j - 1));
        }
        return res;
    }

    /** ##
     * 621. 任务调度器
     */
    public int leastInterval(char[] tasks, int n) {
        Map<Character,Integer> data = new HashMap<>();
        int maxFre = 0;
        for(char ch : tasks) {
            data.put(ch,data.getOrDefault(ch,0) + 1);
            maxFre = Math.max(maxFre,data.get(ch));
        }
        int maxCount = 0;
        for(Map.Entry<Character,Integer> entry : data.entrySet()) {
            if(entry.getValue() == maxFre) maxCount++;
        }
        return Math.max((maxFre - 1)*(n + 1) + maxCount,tasks.length);
    }

    /**
     * 659. 分割数组为连续子序列
     */
    public static boolean isPossible(int[] nums) {
        Map<Integer,Integer> countMap = new HashMap<>();
        for(int num : nums) countMap.put(num, countMap.containsKey(num) ? countMap.get(num) + 1 : 1);
        Map<Integer,Integer> endMap = new HashMap<>();
        for(int num : nums) {
            int count = countMap.getOrDefault(num,0);
            if(count > 0) {
                int pre = endMap.getOrDefault(num - 1,0);
                if(pre > 0) {
                    countMap.put(num,count - 1);
                    endMap.put(num - 1,pre - 1);
                    endMap.put(num,endMap.getOrDefault(num,0) + 1);
                } else {
                    int count1 = countMap.getOrDefault(num + 1,0);
                    int count2 = countMap.getOrDefault(num + 2,0);
                    if(count1 > 0 && count2 > 0) {
                        countMap.put(num, count - 1);
                        countMap.put(num + 1, count1 - 1);
                        countMap.put(num + 2, count2 - 1);
                        endMap.put(num + 2,endMap.getOrDefault(num + 2,0) + 1);
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    // ## 哈希+最小堆
    public boolean isPossible1(int[] nums) {
        Map<Integer, PriorityQueue<Integer>> map = new HashMap<>();
        for (int x : nums) {
            if (!map.containsKey(x)) {
                map.put(x, new PriorityQueue<>());
            }
            if (map.containsKey(x - 1)) {
                int prevLength = map.get(x - 1).poll();
                if (map.get(x - 1).isEmpty()) {
                    map.remove(x - 1);
                }
                map.get(x).offer(prevLength + 1);
            } else {
                map.get(x).offer(1);
            }
        }
        Set<Map.Entry<Integer, PriorityQueue<Integer>>> entrySet = map.entrySet();
        for (Map.Entry<Integer, PriorityQueue<Integer>> entry : entrySet) {
            PriorityQueue<Integer> queue = entry.getValue();
            if (queue.peek() < 3) {
                return false;
            }
        }
        return true;
    }

    /**
     * 204. 计数质数
     */
    public int countPrimes(int n) {
        int[] primes = new int[n];
        Arrays.fill(primes,1);
        int ans = 0;
        for(int i = 2; i < n; i++) {
            if(primes[i] == 1) ans++;
            if((long) i*i < n) {
                for(int j = i*i; j < n; j += i) {
                    primes[j] = 0;
                }
            }
        }
        return ans;
    }

    /** ##
     * 321. 拼接最大数
     */
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        int[] maxSubsequence = new int[k];
        int start = Math.max(0, k - n), end = Math.min(k, m);
        for (int i = start; i <= end; i++) {
            int[] subsequence1 = maxSubsequence(nums1, i);
            int[] subsequence2 = maxSubsequence(nums2, k - i);
            int[] curMaxSubsequence = merge(subsequence1, subsequence2);
            if (compare(curMaxSubsequence, 0, maxSubsequence, 0) > 0) {
                System.arraycopy(curMaxSubsequence, 0, maxSubsequence, 0, k);
            }
        }
        return maxSubsequence;
    }
    public int[] maxSubsequence(int[] nums, int k) {
        int length = nums.length;
        int[] stack = new int[k];
        int top = -1;
        int remain = length - k;
        for (int num : nums) {
            while (top >= 0 && stack[top] < num && remain > 0) {
                top--;
                remain--;
            }
            if (top < k - 1) {
                stack[++top] = num;
            } else {
                remain--;
            }
        }
        return stack;
    }
    public int[] merge(int[] subsequence1, int[] subsequence2) {
        int x = subsequence1.length, y = subsequence2.length;
        if (x == 0) return subsequence2;
        if (y == 0) return subsequence1;
        int mergeLength = x + y;
        int[] merged = new int[mergeLength];
        int index1 = 0, index2 = 0;
        for (int i = 0; i < mergeLength; i++) {
            merged[i] = compare(subsequence1, index1, subsequence2, index2) > 0 ? subsequence1[index1++] : subsequence2[index2++];
        }
        return merged;
    }
    public int compare(int[] subsequence1, int index1, int[] subsequence2, int index2) {
        int x = subsequence1.length, y = subsequence2.length;
        while (index1 < x && index2 < y) {
            int difference = subsequence1[index1] - subsequence2[index2];
            if (difference != 0) return difference;
            index1++;
            index2++;
        }
        return (x - index1) - (y - index2);
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     */
    public static int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1,-1};
        if(nums.length <= 0) return res;
        if(nums.length == 1) {
            if (nums[0] != target) {
                return res;
            } else {
                return new int[]{0,0};
            }
        }
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while(left < right && left >= 0 && right < nums.length) {
            mid = (left + right - 1)/2;
            if(nums[mid] < target) {
                left = mid + 1;
            } else if(nums[mid] > target){
                right = mid - 1;
            } else {
                left = mid;
                break;
            }
        }

        for(int i = left; i >= 0; i--) {
            if(nums[i] == target) {
                res[0] = i;
                res[1] = i;
            }
        }
        int index = res[0];
        while(index < nums.length && index >= 0) {
            if(nums[index] == target) {
                index++;
            } else {
                break;
            }
        }
        if(index != res[0]) res[1] = index - 1;
        return res;
    }

    /** ##
     * 767. 重构字符串
     */
    public String reorganizeString(String S) {
        if(S.length() <= 1) return S;
        int[] data = new int[26];
        int maxCount = 0, length = S.length();
        for(char ch : S.toCharArray()) {
            data[ch - 'a']++;
            maxCount = Math.max(maxCount,data[ch - 'a']);
        }
        if(maxCount > (length + 1) / 2) return "";
        char[] reorganize = new char[length];
        int odd = 1, even = 0;
        for(int i = 0; i < 26; i++) {
            char c = (char) ('a' + i);
            while(data[i] > 0 && data[i] <= length/2 && odd < length) {
                reorganize[odd] = c;
                data[i]--;
                odd += 2;
            }
            while(data[i] > 0) {
                reorganize[even] = c;
                data[i]--;
                even += 2;
            }
        }
        return new String(reorganize);
    }

    /** ##
     * 493. 翻转对
     */
    public int reversePairs(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        return reversePairsRecursive(nums, 0, nums.length - 1);
    }

    public int reversePairsRecursive(int[] nums, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = reversePairsRecursive(nums, left, mid);
            int n2 = reversePairsRecursive(nums, mid + 1, right);
            int ret = n1 + n2;

            // 首先统计下标对的数量
            int i = left;
            int j = mid + 1;
            while (i <= mid) {
                while (j <= right && (long) nums[i] > 2 * (long) nums[j]) {
                    j++;
                }
                ret += j - mid - 1;
                i++;
            }

            // 随后合并两个排序数组
            int[] sorted = new int[right - left + 1];
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = nums[p2++];
                } else if (p2 > right) {
                    sorted[p++] = nums[p1++];
                } else {
                    if (nums[p1] < nums[p2]) {
                        sorted[p++] = nums[p1++];
                    } else {
                        sorted[p++] = nums[p2++];
                    }
                }
            }
            for (int k = 0; k < sorted.length; k++) {
                nums[left + k] = sorted[k];
            }
            return ret;
        }
    }

    /**
     * 976. 三角形的最大周长
     */
    public int largestPerimeter(int[] A) {
        Arrays.sort(A);
        for(int i = A.length - 1; i >= 2; i--) {
            if(A[i - 1] + A[i - 2] > A[i]) {
                return A[i] + A[i - 1] + A[i - 2];
            }
        }
        return 0;
    }

    /**
     * 454. 四数相加 II
     */
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer,Integer> countAB = new HashMap<>();
        for(int a : A) {
            for(int b : B) {
                countAB.put(a + b, countAB.containsKey(a + b) ? countAB.get(a + b) + 1 : 1);
            }
        }
        int ans = 0;
        for(int c : C) {
            for(int d : D) {
                if(countAB.containsKey(-c - d)) {
                    ans += countAB.get(-c - d);
                }
            }
        }
        return ans;
    }

    /**
     * 164. 最大间距
     */
    public int maximumGap(int[] nums) {
        if(nums.length <= 1) return 0;
        Arrays.sort(nums);
        int res = Integer.MIN_VALUE;
        for(int i = 1; i < nums.length; i++) {
            res = Math.max(nums[i] - nums[i - 1], res);
        }
        return res;
    }

    /**
     * 1370. 上升下降字符串
     */
    public String sortString(String s) {
        int[] nums = new int[26];
        for(char ch : s.toCharArray()) {
            nums[ch - 'a'] ++;
        }
        StringBuffer res = new StringBuffer();
        while (res.length() < s.length()) {
            for(int i = 0; i < 26; i++) {
                if(nums[i] != 0) {
                    res.append((char)(i + 'a'));
                    nums[i]--;
                }
            }
            for(int j = 25; j >= 0; j--) {
                if(nums[j] != 0) {
                    res.append((char)(j + 'a'));
                    nums[j]--;
                }
            }
        }
        return res.toString();
    }

    /**
     * 222. 完全二叉树的节点个数
     */
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int level = 0;
        TreeNode node = root;
        while (node.left != null) {
            level++;
            node = node.left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (exists(root, level, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public boolean exists(TreeNode root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0) {
            if ((bits & k) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }
    public int countNodes1(TreeNode root) {
        if(root == null) return 0;
        int count = 1;
        return count + countNodes1(root.left) + countNodes1(root.right);
    }


    /**
     * 452. 用最少数量的箭引爆气球
     */
    public static int findMinArrowShots(int[][] points) {
        if(points.length <= 1) return points.length;
        List<int[]> res = new ArrayList<>();
        Arrays.sort(points,(o1,o2) -> {
            if(o1[0] != o2[0]) return o1[0] - o2[0];
            return o1[1] - o2[1];
        });
        int[] tmp = points[0];
        for(int i = 1; i < points.length; i++) {
            if(points[i][1] < tmp[0] ||points[i][0] > tmp[1]) {
                res.add(tmp);
                tmp = points[i];
            } else if(tmp[0] >= points[i][0] && tmp[0] <= points[i][1] && tmp[1] >= points[i][1]) {
                tmp = new int[]{tmp[0],points[i][1]};
            } else if(tmp[0] <= points[i][0] && tmp[1] >= points[i][0] && tmp[1] <= points[i][1]) {
                tmp = new int[]{points[i][0],tmp[1]};
            } else if(tmp[0] <= points[i][0] && tmp[1] >= points[i][1]){
                tmp = points[i];
            }
        }
        if(res.isEmpty() || res.get(res.size()-1) != tmp) return res.size() + 1;
        return res.size();
    }

    /**
     * 242. 有效的字母异位词
     */
    public boolean isAnagram(String s, String t) {
        if(s.length() != t.length()) return false;
        Map<Character,Integer> tmp1 = new HashMap<>();
        Map<Character,Integer> tmp2 = new HashMap<>();
        for(char ch : s.toCharArray()) {
            tmp1.put(ch,tmp1.containsKey(ch) ? tmp1.get(ch) + 1 : 1);
        }
        for(char ch : t.toCharArray()) {
            tmp2.put(ch,tmp2.containsKey(ch) ? tmp2.get(ch) + 1 : 1);
        }
        for(Map.Entry<Character,Integer> entry : tmp1.entrySet()) {
            if(!tmp2.containsKey(entry.getKey())
                    || !entry.getValue().equals(tmp2.get(entry.getKey()))) return false;
        }
        return true;
    }

    /** ##
     * 148. 排序链表
     */
    public ListNode sortList(ListNode head) {
        if (head == null) {
            return null;
        }
        int length = 0;
        ListNode node = head;
        while (node != null) {
            length++;
            node = node.next;
        }
        ListNode dummyHead = new ListNode(0, head);
        for (int subLength = 1; subLength < length; subLength <<= 1) {
            ListNode prev = dummyHead, curr = dummyHead.next;
            while (curr != null) {
                ListNode head1 = curr;
                for (int i = 1; i < subLength && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode head2 = curr.next;
                curr.next = null;
                curr = head2;
                for (int i = 1; i < subLength && curr != null && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode next = null;
                if (curr != null) {
                    next = curr.next;
                    curr.next = null;
                }
                ListNode merged = merge(head1, head2);
                prev.next = merged;
                while (prev.next != null) {
                    prev = prev.next;
                }
                curr = next;
            }
        }
        return dummyHead.next;
    }
    // 暴力法
    public ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;
    }

    /** ~~
     * 147. 对链表进行插入排序
     */
    public ListNode insertionSortList(ListNode head) {
        if(head == null) return null;
        ListNode ans = new ListNode(0);
        ans.next = head;
        ListNode lastSorted = head, curr = head.next;
        while(curr != null) {
            if(lastSorted.val < curr.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode tmp = ans;
                while(tmp.next.val <= curr.val) {
                    tmp = tmp.next;
                }
                lastSorted.next = curr.next;
                curr.next = tmp.next;
                tmp.next = curr;
            }
            curr = lastSorted.next;
        }
        return ans.next;
    }
    public ListNode insertionSortList1(ListNode head) {
        List<Integer> nums = new ArrayList<>();
        while (head != null) {
            nums.add(head.val);
            head = head.next;
        }
        int[] data = new int[nums.size()];
        for(int i = 0; i < nums.size(); i++) {
            data[i] = nums.get(i);
        }
        Arrays.sort(data);
        ListNode res = new ListNode();
        ListNode tmp = res;
        for(int num : data) {
            tmp.next = new ListNode(num);
            tmp = tmp.next;
        }
        return res.next;
    }

    /**
     * 134. 加油站
     */
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int length = gas.length;
        for(int i = 0; i < length; i++) {
            int currOil = 0, count =0;
            while(count < length) {
                currOil += gas[(i + count) % length];
                if(currOil < cost[(i + count) % length]) {
                    break;
                } else {
                    currOil -= cost[(i + count) % length];
                    count++;
                }
            }
            if(count == length) return i;
        }
        return -1;
    }

    /**
     * 406. 根据身高重建队列
     */
    public static int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (o1, o2) -> {
            if(o1[0] != o2[0]) return o2[0] - o1[0];
            return o1[1] - o2[1];
        });
        List<int[]> ans = new ArrayList<>();
        for(int[] person : people) {
            ans.add(person[1],person);
        }
        return ans.toArray(new int[ans.size()][]);
    }

    /**
     * 5550. 拆炸弹
     */
    public static int[] decrypt(int[] code, int k) {
        int length = code.length;
        int[] res = new int[length];
        if(k == 0) {
            Arrays.fill(res,0);
            return res;
        }
        for(int i = 0; i < length; i++) {
            if(k < 0) {
                int count = -1;
                while (count >= k) {
                    res[i] += code[(i + count + length) % length];
                    count--;
                }
            } else {
                int count = 1;
                while (count <= k) {
                    res[i] += code[(i + count) % length];
                    count++;
                }
            }
        }
        return res;
    }

    /** ##
     * 402. 移掉K位数字
     */
    public String removeKdigits(String num, int k) {
        Deque<Character> deque = new LinkedList<>();
        int length = num.length();
        for (int i = 0; i < length; ++i) {
            char digit = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && deque.peekLast() > digit) {
                deque.pollLast();
                k--;
            }
            deque.offerLast(digit);
        }

        for (int i = 0; i < k; ++i) {
            deque.pollLast();
        }

        StringBuilder ret = new StringBuilder();
        boolean leadingZero = true;
        while (!deque.isEmpty()) {
            char digit = deque.pollFirst();
            if (leadingZero && digit == '0') {
                continue;
            }
            leadingZero = false;
            ret.append(digit);
        }
        return ret.length() == 0 ? "0" : ret.toString();
    }

    /**
     * 1122. 数组的相对排序
     */
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        Map<Integer,Integer> dataMap = new HashMap<>();
        int[] res = new int[arr1.length];
        List<Integer> tmp = new ArrayList<>();
        for(int num : arr2 ) {
            dataMap.put(num, 0);
        }
        for(int num : arr1 ) {
            if(dataMap.containsKey(num)) {
                dataMap.put(num,dataMap.get(num) + 1);
            }else {
                tmp.add(num);
            }
        }
        int index = 0;
        for(int num : arr2) {
            for(int i = index ; i < index + dataMap.get(num); i++) {
                res[i] = num;
            }
            index += dataMap.get(num);
        }
        tmp.sort(Integer::compareTo);
        for(int i = index; i < arr1.length; i++) {
            res[i] = tmp.get(i - index);
        }
        return res;
    }

    /**
     * 328. 奇偶链表
     */
    public ListNode oddEvenList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode first = head;
        ListNode second = head.next;
        ListNode res = new ListNode();
        ListNode tmp = res;
        int index = 1;
        while (first != null ) {
            if(index == 1){
                tmp.next = new ListNode(first.val);
                tmp = tmp.next;
                index = 0;
            }else {
                index = 1;
            }
            first = first.next;
        }
        index = 0;
        while (second != null ) {
            if(index == 0){
                tmp.next = new ListNode(second.val);
                tmp = tmp.next;
                index = 1;
            }else {
                index = 0;
            }
            second = second.next;
        }
        return res.next;
    }
    public ListNode oddEvenList1(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode evenHead = head.next;
        ListNode odd = head, even = evenHead;
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    /**
     * 922. 按奇偶排序数组 II
     */
    public int[] sortArrayByParityII(int[] A) {
        int left = 0, right = 1;
        while(left < A.length && right < A.length) {
            if(A[left]%2 == 1 && A[right]%2 == 0) {
                int tmp = A[left];
                A[left] = A[right];
                A[right] = tmp;
                left += 2;
                right +=2;
            } else if(A[left]%2 == 1) {
                right += 2;
            } else if(A[left]%2 == 0 && A[right]%2 == 1) {
                left += 2;
                right +=2;
            } else {
                left += 2;
            }
        }
        return A;
    }

    /** ##
     * 514. 自由之路
     */
    public int findRotateSteps(String ring, String key) {
        int n = ring.length(), m = key.length();
        List<Integer>[] pos = new List[26];
        for (int i = 0; i < 26; ++i) {
            pos[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; ++i) {
            pos[ring.charAt(i) - 'a'].add(i);
        }
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dp[i], 0x3f3f3f);
        }
        for (int i : pos[key.charAt(0) - 'a']) {
            dp[0][i] = Math.min(i, n - i) + 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j : pos[key.charAt(i) - 'a']) {
                for (int k : pos[key.charAt(i - 1) - 'a']) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][k] + Math.min(Math.abs(j - k), n - Math.abs(j - k)) + 1);
                }
            }
        }
        return Arrays.stream(dp[m - 1]).min().getAsInt();
    }

    /** ##
     * 31. 下一个排列
     */
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }

    /**
     * 122. 买卖股票的最佳时机 II
     */
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        for(int i = 1; i < prices.length; i++) {
            if(prices[i] > prices[i-1]) {
                maxProfit += prices[i] - prices[i-1];
            }
        }
        return maxProfit;
    }

    /**
     * 973. 最接近原点的 K 个点
     */
    public int[][] kClosest(int[][] points, int K) {
        int length = points.length;
        int[][] distance = new int[length][2];
        for(int i = 0;i < length; i++) {
            distance[i][0] = i;
            distance[i][1] = (int) (Math.pow(points[i][0],2)+ Math.pow(points[i][1], 2));
        }
        Arrays.sort(distance,(o1,o2) -> {
            if(o1[1] == o2[1]) return o1[0] - o2[0];
            return o1[1] - o2[1];
        });
        int[][] res = new int[K][2];
        for(int i = 0; i < K; i++) {
            res[i] = points[distance[i][0]];
        }
        return res;
    }

    /** ##
     * 140. 单词拆分 II
     */
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<Integer, List<List<String>>> map = new HashMap<>();
        List<List<String>> wordBreaks = backtrack(s, s.length(), new HashSet<String>(wordDict), 0, map);
        List<String> breakList = new LinkedList<>();
        for (List<String> wordBreak : wordBreaks) {
            breakList.add(String.join(" ", wordBreak));
        }
        return breakList;
    }

    public List<List<String>> backtrack(String s, int length, Set<String> wordSet, int index, Map<Integer, List<List<String>>> map) {
        if (!map.containsKey(index)) {
            List<List<String>> wordBreaks = new LinkedList<>();
            if (index == length) {
                wordBreaks.add(new LinkedList<>());
            }
            for (int i = index + 1; i <= length; i++) {
                String word = s.substring(index, i);
                if (wordSet.contains(word)) {
                    List<List<String>> nextWordBreaks = backtrack(s, length, wordSet, i, map);
                    for (List<String> nextWordBreak : nextWordBreaks) {
                        LinkedList<String> wordBreak = new LinkedList<>(nextWordBreak);
                        wordBreak.offerFirst(word);
                        wordBreaks.add(wordBreak);
                    }
                }
            }
            map.put(index, wordBreaks);
        }
        return map.get(index);
    }
    /** ##
     * 327. 区间和的个数
     */
    public static int countRangeSum(int[] nums, int lower, int upper) {
        long s = 0;
        long[] sum = new long[nums.length + 1];
        for (int i = 0; i < nums.length; ++i) {
            s += nums[i];
            sum[i + 1] = s;
        }
        return countRangeSumRecursive(sum, lower, upper, 0, sum.length - 1);
    }

    public static int countRangeSumRecursive(long[] sum, int lower, int upper, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = countRangeSumRecursive(sum, lower, upper, left, mid);
            int n2 = countRangeSumRecursive(sum, lower, upper, mid + 1, right);
            int ret = n1 + n2;

            // 首先统计下标对的数量
            int i = left;
            int l = mid + 1;
            int r = mid + 1;
            while (i <= mid) {
                while (l <= right && sum[l] - sum[i] < lower) {
                    l++;
                }
                while (r <= right && sum[r] - sum[i] <= upper) {
                    r++;
                }
                ret += r - l;
                i++;
            }

            // 随后合并两个排序数组
            int[] sorted = new int[right - left + 1];
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = (int) sum[p2++];
                } else if (p2 > right) {
                    sorted[p++] = (int) sum[p1++];
                } else {
                    if (sum[p1] < sum[p2]) {
                        sorted[p++] = (int) sum[p1++];
                    } else {
                        sorted[p++] = (int) sum[p2++];
                    }
                }
            }
            for (int j = 0; j < sorted.length; j++) {
                sum[left + j] = sorted[j];
            }
            return ret;
        }
    }
    // 暴力法
    public static int countRangeSum1(int[] nums, int lower, int upper) {
        List<int[]> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] <= upper && nums[i] >= lower) {
                res.add(new int[]{i,i});
            }
            double tempSum = nums[i];
            for(int j = i + 1; j < nums.length; j++) {
                if(tempSum + nums[j] <= upper && tempSum + nums[j] >= lower) {
                    res.add(new int[]{i,j});
                }
                tempSum += nums[j];
            }
        }
        return res.size();
    }

    /**
     * 1024. 视频拼接
     */
    public int videoStitching(int[][] clips, int T) {
        int[] dp = new int[T + 1];
        Arrays.fill(dp, Integer.MAX_VALUE - 1);
        dp[0] = 0;
        for (int i = 1; i <= T; i++) {
            for (int[] clip : clips) {
                if (clip[0] < i && i <= clip[1]) {
                    dp[i] = Math.min(dp[i], dp[clip[0]] + 1);
                }
            }
        }
        return dp[T] == Integer.MAX_VALUE - 1 ? -1 : dp[T];
    }
}

/**
 * 304. 二维区域和检索 - 矩阵不可变
 */
class NumMatrix {
    int[][] sums;
    public NumMatrix(int[][] matrix) {
        int row = matrix.length;
        if(row > 0) {
            int col = matrix[0].length;
            sums = new int[row + 1][col + 1];
            for(int i = 0; i < row; i++) {
                for(int j = 0; j < col; j++) {
                    sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j] - sums[i][j] + matrix[i][j];
                }
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
    }
}

