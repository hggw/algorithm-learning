package LeetCodeExcise;

import java.util.*;

public class LeetCodeEasySolution {
    public static void main(String[] args) {
        int data = 0;
        List<Integer> res = getRow(data);
    }

    /**
     * 119. 杨辉三角 II
     */
    public static List<Integer> getRow(int rowIndex) {
        int[][] res = new int[rowIndex+1][rowIndex+1];
        res[0][0] = 1;
        if(rowIndex >= 1) {
            res[1][0] = 1;
            res[1][1] = 1;
        }
        for(int i = 2; i < rowIndex+1; i++) {
            res[i][0] = 1;
            for(int j = 1; j < i; j++) {
                res[i][j] = res[i-1][j-1] + res[i-1][j];
            }
            res[i][i] = 1;
        }
        List<Integer> ans = new ArrayList<>();
        for(int i = 0; i < rowIndex+1; i++) {
            ans.add(res[rowIndex][i]);
        }
        return ans;
    }
    /**
     * 1331. 数组序号转换
     */
    public static int[] arrayRankTransform(int[] arr) {
        int[] tmp = arr.clone();
        Arrays.sort(tmp);
        Map<Integer,Integer> data = new HashMap<>();
        int index = 1;
        for(int num : tmp) {
            if(!data.containsKey(num)) {
                data.put(num,index);
                index++;
            }
        }for(int i = 0; i < arr.length; i++) {
            arr[i] = data.get(arr[i]);
        }
        return arr;
    }

    /**
     * 917. 仅仅反转字母
     */
    public static String reverseOnlyLetters(String S) {
        StringBuffer ans = new StringBuffer();
        int j = S.length() - 1;
        for(int i = 0; i < S.length(); i++) {
            if(Character.isLetter(S.charAt(i))) {
                while(!Character.isLetter(S.charAt(j))) {
                    j--;
                }
                ans.append(S.charAt(j--));
            } else {
                ans.append(S.charAt(i));
            }
        }
        return ans.toString();
    }

    /**
     * 500. 键盘行
     */
    public String[] findWords1(String[] words) {
        char[] first = new char[]{'q','w','e','r','t','y','u','i','o','p','Q','W','E','R','T','Y','U','I','O','P'};
        char[] second = new char[]{'a','s','d','f','g','h','j','k','l','A','S','D','F','G','H','J','K','L'};
        char[] third = new char[]{'z','x','c','v','b','n','m','Z','X','C','V','B','N','M'};
        Set<Character> tmp1 = new HashSet<>();
        Set<Character> tmp2 = new HashSet<>();
        Set<Character> tmp3 = new HashSet<>();
        for(char ch : first) {
            tmp1.add(ch);
        }
        for(char ch : second) {
            tmp2.add(ch);
        }
        for(char ch : third) {
            tmp3.add(ch);
        }
        List<String> res = new ArrayList<>();
        for(String word : words) {
            int count1 = 0,count2 = 0, count3 = 0;
            for(char ch : word.toCharArray()) {
                if(tmp1.contains(ch)) {
                    count1++;
                } else if(tmp2.contains(ch)) {
                    count2++;
                } else {
                    count3++;
                }
            }
            if(count1 == word.length() || count2 == word.length() || count3 == word.length()) res.add(word);
        }
        return res.toArray(new String[res.size()]);
    }
    public String[] findWords(String[] words) {
        if (words == null) return null;
        List<String> ans = new ArrayList<>();
        String[] lines = new String[] {"qwertyuiop", "asdfghjkl", "zxcvbnm"};
        for (String word : words) {
            String str = word.toLowerCase();
            for(String string : lines) {
                if(string.indexOf(str.charAt(0)) > -1) {
                    int count = 1;
                    for(int i = 1; i < str.length(); i++) {
                        if(string.indexOf(str.charAt(i)) > -1) {
                            count++;
                        }
                    }
                    if(count == word.length()) {
                        ans.add(word);
                    } else {
                        break;
                    }

                }
            }
        }
        return ans.toArray(new String[ans.size()]);
    }
    private boolean judge(String word,String[] lines) {
        boolean ok = true;
        String find = null;

        // 先用word首字符确定属于哪一行
        for (String line : lines) {
            if (line.indexOf(word.charAt(0)) > -1) {
                find = line;
                break;
            }
        }

        if (find == null) {
            ok = false;
            return ok;
        }

        // 判断word字符串所有字符是否都属于同一行
        for (int i = 1;i < word.length();i++) {
            if (find.indexOf(word.charAt(i)) < 0) {
                ok = false;
                break;
            }
        }

        return ok;
    }

    /**
     * 228. 汇总区间
     */
    public static List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if(nums.length <= 0) return res;
        if(nums.length <= 1) {
            res.add(String.valueOf(nums[0]));
            return res;
        }
        int left = nums[0],right = nums[0];
        for(int i = 1; i < nums.length; i++) {
            if((long)nums[i] - (long)nums[i - 1] > 1) {
                if(left == right) {
                    res.add(String.valueOf(left));
                } else {
                    res.add(left + "->" + right);
                }
                left = nums[i];
                right = nums[i];
            } else {
                right++;
            }
        }
        if(left == right) {
            res.add(String.valueOf(left));
        } else {
            res.add(left + "->" + right);
        }
        return res;
    }

    /** ##
     * 953. 验证外星语词典
     */
    public boolean isAlienSorted(String[] words, String order) {
        int[] index = new int[26];
        for (int i = 0; i < order.length(); ++i)
            index[order.charAt(i) - 'a'] = i;

        search: for (int i = 0; i < words.length - 1; ++i) {
            String word1 = words[i];
            String word2 = words[i+1];

            // Find the first difference word1[k] != word2[k].
            for (int k = 0; k < Math.min(word1.length(), word2.length()); ++k) {
                if (word1.charAt(k) != word2.charAt(k)) {
                    // If they compare badly, it's not sorted.
                    if (index[word1.charAt(k) - 'a'] > index[word2.charAt(k) - 'a'])
                        return false;
                    continue search;
                }
            }

            // If we didn't find a first difference, the
            // words are like ("app", "apple").
            if (word1.length() > word2.length())
                return false;
        }

        return true;
    }

    /**
     * 1496. 判断路径是否相交
     */
    public static boolean isPathCrossing(String path) {
        Set<String> process = new HashSet<>();
        process.add("00");
        int heng = 0, shu = 0;
        for(char ch : path.toCharArray()) {
            heng += ch == 'E' ? 1 : ch == 'W' ? -1 : 0;
            shu += ch == 'N' ? 1 : ch == 'S' ? -1 : 0;
            if(process.contains(heng + Integer.toString(shu))) return true;
            process.add(heng + Integer.toString(shu));
        }
        return false;
    }

    /**
     * 1576. 替换所有的问号
     */
    public String modifyString(String s) {
        char[] chars = s.toCharArray();
        for(int i = 0;i < chars.length; i++) {
            if (chars[i] == '?') {
                char ahead = i == 0 ? ' ' : chars[i - 1];
                char after = i == chars.length - 1 ? ' ' : chars[i + 1];
                char tmp = 'a';
                while (tmp == after || tmp == ahead) {
                    tmp++;
                }
                chars[i] = tmp;
            }
        }
        return new String(chars);
    }

    /**
     * 575. 分糖果
     */
    public int distributeCandies(int[] candies) {
        Set<Integer> candy = new HashSet<>();
        for(int i : candies) {
            candy.add(i);
        }
        return Math.min(candy.size(), candies.length / 2);
    }

    /**
     * 83. 删除排序链表中的重复元素
     */
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) return null;
        Set<Integer> res = new HashSet<>();
        res.add(head.val);
        ListNode tmp = head;
        while(tmp.next != null) {
            if(!res.contains(tmp.next.val)) {
                res.add(tmp.next.val);
                tmp = tmp.next;
            } else if(tmp.next.next != null) {
                tmp.next = tmp.next.next;
            } else {
                break;
            }
        }
        if(tmp.next == null || res.contains(tmp.next.val)) {
            tmp.next = null;
        }
        return head;
    }
    public ListNode deleteDuplicates1(ListNode head) {
        ListNode current = head;
        while (current != null && current.next != null) {
            if (current.next.val == current.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return head;
    }

    /**
     * 111. 二叉树的最小深度
     */
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        if(root.left == null && root.right == null) {
            return 1;
        }
        int res = Integer.MAX_VALUE;
        if(root.left != null) res = Math.min(minDepth(root.left),res);
        if(root.right != null) res = Math.min(minDepth(root.right),res);
        return res + 1;
    }

    /**
     * 101. 对称二叉树
     */
    // 递归法
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        return isSame(root.left,root.right);
    }
    public boolean isSame(TreeNode nodeLeft, TreeNode nodeRight) {
        if(nodeLeft == null || nodeRight == null) return nodeLeft == nodeRight;
        if(nodeLeft.val != nodeRight.val) return false;
        return isSame(nodeLeft.left,nodeRight.right) && isSame(nodeLeft.right,nodeRight.left);
    }
    //迭代法
    public boolean isSymmetric1(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode u, TreeNode v) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(u);
        q.offer(v);
        while (!q.isEmpty()) {
            u = q.poll();
            v = q.poll();
            if (u == null && v == null) {
                continue;
            }
            if ((u == null || v == null) || (u.val != v.val)) {
                return false;
            }

            q.offer(u.left);
            q.offer(v.right);

            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
    }

    /**
     * 415. 字符串相加
     */
    public static String addStrings(String num1, String num2) {
        int indexOne = num1.length() - 1, indexTwo = num2.length() - 1;
        StringBuilder res = new StringBuilder();
        int rest = 0;
        while(indexOne >= 0 || indexTwo >= 0) {
            int x = indexOne >= 0 ? num1.charAt(indexOne) - '0' : 0;
            int y = indexTwo >= 0 ? num2.charAt(indexTwo) - '0' : 0;
            int result = x + y + rest;
            res.append(result%10);
            rest = result/10;
            indexOne--;
            indexTwo--;
        }
        if(rest != 0) res.append(rest);
        return res.reverse().toString();
    }

    /**
     * 112. 路径总和
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        if(root.left == null && root.right == null) return sum == root.val;
        return hasPathSum(root.left,sum - root.val) || hasPathSum(root.right,sum - root.val);
    }

    /**
     * 69. x 的平方根
     */
    public int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    /**
     * 268. 丢失的数字
     */
    public int missingNumber(int[] nums) {
        int index = 0;
        while(index < nums.length) {
            if(nums[index] != index && nums[index] < nums.length) {
                int tmp = nums[index];
                nums[index] = nums[nums[index]];
                nums[tmp] = tmp;
            } else {
                index++;
            }
        }
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != i) return i;
        }
        return nums.length;
    }
    public int missingNumber1(int[] nums) {
        int res = 0;
        for(int i = 0; i < nums.length; i++) {
            res ^= i;
            res ^= nums[i];
        }
        return res^nums.length;
    }

    /**
     * 125. 验证回文串
     */
    public static boolean isPalindrome(String s) {
        StringBuffer res = new StringBuffer();
        int length = s.length();
        for (int i = 0; i < length; i++) {
            char ch = s.charAt(i);
            if (Character.isLetterOrDigit(ch)) {
                res.append(Character.toLowerCase(ch));
            }
        }
        return res.toString().equals(res.reverse().toString());
    }

    /**
     * 141. 环形链表
     */
    public boolean hasCycle(ListNode head) {
        if(head == null || head.next == null) return false;
        ListNode slow = head;
        ListNode fast = head.next;
        while(slow != fast) {
            if(fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    /**
     * 448. 找到所有数组中消失的数字
     */
    public static List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int length = nums.length;
        int index = 0;
        while(index < length ) {
            if(nums[index] != (index + 1) && nums[nums[index] - 1] != nums[index]){
                int tmp = nums[index];
                nums[index] = nums[nums[index] - 1];
                nums[tmp - 1] = tmp;
            }else {
                index++;
            }
        }
        for(int i = 0; i < length; i++){
            if(nums[i] != (i + 1)) res.add(i + 1);
        }
        return res;
    }

    /**
     *
     */
    public int minCostToMoveChips(int[] position) {
        int odd = 0, even = 0;
        for(int num : position) {
            odd += num % 2 == 1 ? 1 : 0;
            even += num % 2 == 0 ? 1 : 0;
        }
        return Math.min(even, odd);
    }

    /** ##
     * 20. 有效的括号
     */
    public boolean isValid(String s) {
        int length = s.length();
        if(length % 2 == 1) return false;
        Map<Character,Character> pairs = new HashMap<Character,Character>(){{
            put(')','(');
            put(']','[');
            put('}','{');
        }};
        Deque<Character> stack = new LinkedList<>();
        for(char ch : s.toCharArray()) {
            if(pairs.containsKey(ch)) {
                if(stack.isEmpty() || stack.peek() != pairs.get(ch)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }

    /**
     * 1304. 和为零的N个唯一整数
     */
    public static int[] sumZero(int n) {
        int[] res = new int[n];
        int length = n / 2;
        for(int i = 0; i < n; i += 2) {
            res[i] = length;
            if(i+1 <= n-1) res[i + 1] = length * (-1);
            length--;
        }
        if(n%2 == 1) res[n - 1] = 0;
        return res;
    }

    /**
     * 108. 将有序数组转换为二叉搜索树
     */
    public static TreeNode sortedArrayToBST(int[] nums) {
        int length = nums.length;
        int left = length / 2, right = length / 2;
        if(length == 0) return null;
        if(length == 1) return new TreeNode(nums[0]);
        TreeNode root = new TreeNode(nums[length / 2]);
        root.left = sortedArrayToBST(Arrays.copyOfRange(nums,0,left));
        root.right = sortedArrayToBST(Arrays.copyOfRange(nums,right+1,length));
        return root;
    }
    /**
     * 292. Nim 游戏
     */
    public static boolean canWinNim(int n) {
        return n % 4 != 0;
    }

    /**
     * 852. 山脉数组的峰顶索引
     */
    public int peakIndexInMountainArray(int[] arr) {
        int left = 0, right = arr.length - 1;
        while(left < right) {
            int mid = (left + right) / 2;
            if(arr[mid] < arr[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    /**
     * 944. 删列造序
     */
    public int minDeletionSize(String[] A) {
        if(A.length < 2) return 0;
        int length = A.length;
        int strLen = A[0].length();
        int count = 0;
        for(int i = 0; i < strLen; i++) {
            int left = 0, right = 1;
            while(right < length) {
                if(A[left].charAt(i) <= A[right].charAt(i)) {
                    left++;
                    right++;
                } else {
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    /**
     * 557. 反转字符串中的单词 III
     */
    public static String reverseWords(String s) {
        StringBuilder res = new StringBuilder();
        res.append(s);
        String[] ans = res.reverse().toString().split(" ");
        StringBuilder result = new StringBuilder();
        for(int i = ans.length - 1; i >= 0; i--) {
            result.append(ans[i]).append(" ");
        }
        return result.toString().trim();
    }

    /**
     * 344. 反转字符串
     */
    public void reverseString(char[] s) {
        int length = s.length;
        for(int i = 0; i < length/2; i++) {
            char tmp = s[length - 1 - i];
            s[length - 1 - i] = s[i];
            s[i] = tmp;
        }
    }

    /** ??本题问的是调整的人数，如要求调整的次数，要怎么解？
     * 1051. 高度检查器
     */
    public int heightChecker(int[] heights) {
        int[] arr = new int[101];
        for(int height : heights) {
            arr[height]++;
        }
        int count = 0;
        for(int i = 1, j = 0; i < arr.length; i++) {
            while(arr[i]-- > 0) {
                if(heights[j++] != i) count++;
            }
        }
        return count;
    }
    // 求解调整的次数 ？？
    public static int heightChangeTimes(int[] heights) {
        int[] tmp = new int[heights.length];
        for(int i = 0 ; i < heights.length; i++) tmp[i] = heights[i];
        Arrays.sort(tmp);
        int count = 0;
        for(int i = 0; i < heights.length; i++) {
            if(tmp[i] != heights[i]) {
                count++;
                int tmp1 = heights[i];
                heights[i] = tmp[i];
                heights[Arrays.binarySearch(tmp,tmp1)] = tmp1;
            }
        }
        return count;
    }

    /**
     * 590. N叉树的后序遍历
     */
    public List<Integer> postorder(Node root) {
        List<Integer> res = new ArrayList<>();
        if(root == null)  return res;
        List<Node> tmp  = root.children;
        for(Node node : tmp) {
            List<Integer> tmp1 = postorder(node);
            if(tmp1 != null || tmp1.size() != 0) {
                res.addAll(tmp1);
            }
        }
        res.add(root.val);
        return res;
    }

    /**
     * 1252. 奇数值单元格的数目
     */
    public static int oddCells(int n, int m, int[][] indices) {
        int count = 0;
        Map<Integer,Integer> rowMap = new HashMap<>();
        Map<Integer,Integer> colMap = new HashMap<>();
        for (int[] index : indices) {
            rowMap.put(index[0], rowMap.containsKey(index[0]) ? rowMap.get(index[0]) + 1 : 1);
            colMap.put(index[1], colMap.containsKey(index[1]) ? colMap.get(index[1]) + 1 : 1);
        }
        int evenCount = 0;
        for(Map.Entry<Integer,Integer> entity : rowMap.entrySet()) {
            count += entity.getValue() % 2 == 1 ? m : 0;
            evenCount += entity.getValue() % 2 == 0 ? 1 : 0;
        }
        for(Map.Entry<Integer,Integer> entity : colMap.entrySet()) {
            count += entity.getValue() % 2 == 1 ? n - rowMap.size() + evenCount - (rowMap.size() - evenCount) : 0;
        }
        return Math.max(count, 0);
    }
    // 暴力法
    public static int oddCells1(int n, int m, int[][] indices) {
        int[][] res = new int[n][m];
        for (int[] index : indices) {
            for (int j = 0; j < m; j++) {
                res[index[0]][j]++;
            }
            for (int k = 0; k < n; k++) {
                res[k][index[1]]++;
            }
        }
        int count = 0;
        for(int[] x : res) {
            for(int num : x) {
                count += num % 2 == 1 ? 1 : 0;
            }
        }
        return count;
    }

    /**
     * 1619. 删除某些元素后的数组均值
     */
    public double trimMean(int[] arr) {
        Arrays.sort(arr);
        int start = (int) (arr.length * 0.05);
        double sum = 0;
        for(int i = start; i < arr.length - start; i++) {
            sum += arr[i];
        }
        return sum/(arr.length - 2 * start);
    }

    /**
     * 1534. 统计好三元组
     */
    public int countGoodTriplets(int[] arr, int a, int b, int c) {
        int count = 0;
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                for(int k = j+1; k <arr.length; k++) {
                    if(Math.abs(arr[i] - arr[j]) <= a && Math.abs(arr[j] - arr[k]) <= b && Math.abs(arr[i] - arr[k]) <= c) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    /**
     * 1021. 删除最外层的括号
     */
    public static String removeOuterParentheses(String S) {
        List<String> tmp = new ArrayList<>();
        int left = 0, right = 0;
        int index = 0;
        for(int i = 0; i < S.length(); i++) {
            if(left == right && left * right != 0) {
                tmp.add(S.substring(index,i));
                left = 0;
                right = 0;
                index = i;
            }
            if(S.charAt(i) == '(') {
                left++;
            }
            if(S.charAt(i) == ')') {
                right++;
            }
        }
        if(left == right && left * right != 0) {
            tmp.add(S.substring(index));
        }
        StringBuilder ans = new StringBuilder();
        for(String str : tmp) {
            ans.append(str, 1, str.length()-1);
        }
        return ans.toString();
    }

    /**
     * 1356. 根据数字二进制下 1 的数目排序
     */
    public int[] sortByBits(int[] arr) {
        int[][] res = new int[arr.length][2];
        for(int i = 0; i < arr.length; i++ ) {
            res[i][0] = arr[i];
            int count = 0;
            while (arr[i] != 0) {
                if((arr[i] & 1) == 1) {
                    count++;
                }
                arr[i] = arr[i] >> 1;
            }
            res[i][1] = count;
        }
        Arrays.sort(res,(o1, o2) -> {
            if(o1[1] == o2[1]) return o1[0]-o2[0];
            return o1[1]-o2[1];
        });
        int[] ans = new int[arr.length];
        for(int i = 0; i < arr.length; i++) {
            ans[i] = res[i][0];
        }
        return ans;
    }

    /**
     * 938. 二叉搜索树的范围和
     */
    public int rangeSumBST(TreeNode root, int L, int R) {
        if(root == null) {
            return 0;
        }
        int result = 0;
        if(root.val <= R && root.val > L) {
            result += root.val + rangeSumBST(root.left,L,R) + rangeSumBST(root.right,L,R);
        }
        if(root.val > R) {
            result += rangeSumBST(root.left,L,R);
        }
        if(root.val < L) {
            result += rangeSumBST(root.right,L,R);
        }
        return result;
    }
    public int countSum1(TreeNode node, int target) {
        if(node != null) {
            return 0;
        }
        int res = 0;
        if(node.val > target) {
            res += node.val + countSum1(node.left,target);
        }else {
            res += countSum1(node.right,target);
        }
        return res;
    }
    public int countSum2(TreeNode node, int target) {
        if(node != null) {
            return 0;
        }
        int res = 0;
        if(node.val < target) {
            res += node.val + countSum2(node.right,target);
        }else {
            res += countSum2(node.left,target);
        }
        return res;
    }

    /**
     * 1436. 旅行终点站
     */
    public String destCity(List<List<String>> paths) {
        Map<String,String> cityMap = new HashMap<>();
        for (List<String> city : paths) {
            cityMap.put(city.get(0),city.get(1));
        }
        for (List<String> city : paths) {
            int count = 0;
            String startCity = city.get(0);
            while (cityMap.containsKey(startCity)) {
                count++;
                startCity = cityMap.get(startCity);
            }
            if(count == cityMap.size()) {
                return startCity;
            }
        }
        return null;
    }

    /**
     * 226. 翻转二叉树
     */
    public TreeNode invertTree(TreeNode root) {
        if(root == null){
            return null;
        }
        TreeNode res = new TreeNode(root.val);
        res.left = invertTree(root.right);
        res.right = invertTree(root.left);
        return res;
    }

    /**
     * 617. 合并二叉树
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null || t2 ==null) {
            return t1 != null ? t1 : t2;
        }
        TreeNode ans = new TreeNode(t1.val + t2.val);
        ans.left = mergeTrees(t1.left,t2.left);
        ans.right = mergeTrees(t1.right,t2.right);
        return ans;
    }

    /**
     * 1221. 分割平衡字符串
     */
    public static int balancedStringSplit(String s) {
        int left = 0 ,right = 0;
        int count = 0;
        for(char ch : s.toCharArray()) {
            if(left == right && left *right != 0) {
                count++;
                left = 0;
                right = 0;
            }
            if(ch == 'L') {
                left++;
            }else {
                right++;
            }
        }
        return count + 1;
    }

    /**
     * LCP 17. 速算机器人
     */
    public int calculate(String s) {
        int x = 1,y = 0;
        for(char ch : s.toCharArray()){
            x = ch == 'A' ? 2 * x + y : x;
            y = ch == 'B' ? 2 * y + x : y;
        }
        return x + y;
    }
    public int calculate1(String s) {
        return 1 << s.length();
    }

    /**
     * 1572. 矩阵对角线元素的和
     */
    public int diagonalSum(int[][] mat) {
        int row = mat.length;
        int diagSum = 0;
        for(int i = 0; i < row; i++) {
            diagSum += mat[i][i] + mat[i][row-1-i];
        }
        return diagSum - mat[(row - 1) / 2][(row - 1) / 2] * (row & 1);
    }

    /**
     * 1588. 所有奇数长度子数组的和
     */
    public int sumOddLengthSubarrays(int[] arr) {
        int totalSum = 0;
        for(int i = 0; i < arr.length; i++) {
            int left = i + 1 , right = arr.length - i;
            int leftOdd = (left + 1) / 2, rightOdd = (right + 1) / 2;
            int leftEven = left / 2 ,rightEven = right / 2;
            totalSum += (leftEven * rightEven + leftOdd * rightOdd) * arr[i];
        }
        return totalSum;
    }

    /**
     * 1266. 访问所有点的最小时间
     */
    public int minTimeToVisitAllPoints(int[][] points) {
        int row = points.length;
        int totalTime = 0;
        for(int i = 1; i < row; i++) {
            totalTime += Math.max(Math.abs(points[i][0] - points[i-1][0]),Math.abs(points[i][1] - points[i-1][1]));
        }
        return totalTime;
    }

    /**
     * 1431. 拥有最多糖果的孩子
     */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List<Boolean> res = new ArrayList<>();
        int maxCandies = 0;
        for(int candy : candies){
            maxCandies = candy > maxCandies ? candy : maxCandies;
        }
        for(int candy : candies){
            res.add((candy + extraCandies) >= maxCandies);
        }
        return res;
    }

    /**
     * 771. 宝石与石头
     */
    public int numJewelsInStones(String J, String S) {
        Map<Character,Integer> tmp1 = new HashMap<>();
        Map<Character,Integer> tmp2 = new HashMap<>();
        for(char character : J.toCharArray()) {
            if(tmp1.containsKey(character)){
                tmp1.put(character,tmp1.get(character)+1);
            } else {
                tmp1.put(character,1);
            }
        }
        for(char character : S.toCharArray()) {
            if(tmp2.containsKey(character)){
                tmp2.put(character,tmp2.get(character)+1);
            } else {
                tmp2.put(character,1);
            }
        }
        int res = 0;
        for(Map.Entry<Character,Integer> entry : tmp1.entrySet()) {
            if(tmp2.containsKey(entry.getKey())) {
                res += tmp2.get(entry.getKey());
            }
        }
        return res;
    }

    /** 移窗法
     * 1640. 能否连接形成数组
     */
    public boolean canFormArray(int[] arr, int[][] pieces) {
        Map<Integer,int[]> dataMap = new HashMap<>();
        for (int[] piece : pieces) {
            dataMap.put(piece[0], piece);
        }
        for (int i = 0; i < arr.length; ) {
            if(dataMap.containsKey(arr[i])){
                int[] curr = dataMap.get(arr[i]);
                for (int number : curr ) {
                    if(number == arr[i]) {
                        i++;
                    }else {
                        return false;
                    }
                }
            }else {
                return false;
            }
        }
        return true;
    }

    /**
     * 414. 第三大的数
     */
    public static int thirdMax(int[] nums) {
        long first = Long.MIN_VALUE,second = Long.MIN_VALUE, third = Long.MIN_VALUE;
        for(int number : nums){
            if(number > first){
                third = second;
                second = first;
                first = number;
            } else if(number > second && number < first){
                third = second;
                second = number;
            } else if(number > third && number < second){
                third = number;
            }
        }
        return Math.toIntExact(third == Long.MIN_VALUE ? first : third);
    }

    /** ##
     * 404. 左叶子之和
     */
    public int sumOfLeftLeaves(TreeNode root) {
        return root != null ? dfs(root) : 0;
    }

    public int dfs(TreeNode node) {
        int ans = 0;
        if (node.left != null) {
            ans += isLeafNode(node.left) ? node.left.val : dfs(node.left);
        }
        if (node.right != null && !isLeafNode(node.right)) {
            ans += dfs(node.right);
        }
        return ans;
    }

    public boolean isLeafNode(TreeNode node) {
        return node.left == null && node.right == null;
    }

    /**
     * 1154. 一年中的第几天
     */
    public static int dayOfYear(String date) {
        int days = 0;
        int year = Integer.parseInt(date.substring(0,4));
        int month = Integer.parseInt(date.substring(5,7));
        int day = Integer.parseInt(date.substring(8,10));
        int[] data = new int[]{31,28,31,30,31,30,31,31,30,31,30,31};
        if(year%4==0 && year%100!=0){
            for(int i = 0;i<month-1;i++){
                days += data[i];
                if(i==1){
                    days++;
                }
            }
            days += day;
        }else {
            for(int i = 0;i<month-1;i++){
                days += data[i];
            }
            days += day;
        }
        return days;
    }

    /**
     * 845. 数组中的最长山脉
     */
    public int longestMountain(int[] A) {
        int n = A.length;
        if(n<3){
            return 0;
        }
        int[] left = new int[n];
        for(int i = 1; i < n-1; i++){
            left[i] = A[i] > A[i - 1] ? left[i - 1] + 1:0;
        }
        int[] right = new int[n];
        for(int i = n - 2;i > 0;i--){
            right[i] = A[i] > A[i + 1] ? right[i + 1]+1:0;
        }
        int ans = 0;
        for(int i = 0; i < n; i++){
            if(left[i]*right[i]!=0){
                ans = Math.max(ans,left[i]+right[i]+1);
            }
        }
        return ans;
    }

    /**
     * 941. 有效的山脉数组
     */
    public static boolean validMountainArray(int[] A) {
        int length = A.length;
        if(length<=2){
            return false;
        }
        int left = 0,right = 0;
        for(int i = 1;i<length;i++){
            if(A[i]>A[i-1]){
                left++;
                right++;
            }else{
                break;
            }
        }
        for(int i = left+1;i<length;i++){
            if(A[i]<A[i-1]){
                right++;
            }else{
                break;
            }
        }
        return left != 0 && left != length - 1 && right != 0 && right == length - 1;
    }

    /**
     * 349. 两个数组的交集
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> res1 = new HashSet<>();
        Set<Integer> res2 = new HashSet<>();
        for(int number1:nums1){
            res1.add(number1);
        }
        for(int number2:nums2){
            res2.add(number2);
        }
        Set<Integer> res = new HashSet<>();
        if(res1.size()>res2.size()){
            for(Integer x :res2){
                if(res1.contains(x)){
                    res.add(x);
                }
            }
        }else {
            for(Integer x :res1){
                if(res2.contains(x)){
                    res.add(x);
                }
            }
        }
        int[] ans = new int[res.size()];
        int count = 0;
        for(Integer x:res){
            ans[count] = x;
            count++;
        }
        return ans;
    }

    /**
     * 100. 相同的树
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null && q==null){
            return true;
        }
        if(p==null){
            return false;
        }
        if(q==null){
            return false;
        }
        if(p.val!=q.val){
            return false;
        }
        return isSameTree(p.left,q.left) & isSameTree(p.right,q.right);
    }

    /**
     * 985. 查询后的偶数和
     */
    public static int[] sumEvenAfterQueries(int[] A, int[][] queries) {
        int row = queries.length;
        int[] res = new int[row];
        int ouSum = 0;
        Set<Integer> ouSet = new HashSet<>();
        for(int i=0;i<A.length;i++){
            if(A[i]%2==0){
                ouSum += A[i];
                ouSet.add(i);
            }
        }
        for(int i = 0;i<row;i++){
            A[queries[i][1]] += queries[i][0];
            if(A[queries[i][1]]%2==0 ){
                if(!ouSet.contains(queries[i][1])){
                    ouSet.add(queries[i][1]);
                    ouSum += A[queries[i][1]];
                }else{
                    ouSum += queries[i][0];
                }
            }
            if(A[queries[i][1]]%2!=0 && ouSet.contains(queries[i][1])){
                ouSet.remove(queries[i][1]);
                ouSum -= A[queries[i][1]]-queries[i][0];
            }
            res[i] = ouSum;
        }
        return res;
    }

    /**
     * 1556. 千位分隔数
     */
    public static String thousandSeparator(int n) {
        if(n<1000){
            return String.valueOf(n);
        }
        StringBuilder tmp = new StringBuilder();
        int count = 0;
        while(n!=0){
            tmp.append(n%10);
            count++;
            if(count%3==0){
                tmp.append(".");
            }
            n /=10;
        }
        if(tmp.length()%4==0){
            tmp.deleteCharAt(tmp.length()-1);
        }
        return tmp.reverse().toString();
    }

    /**
     * 1185. 一周中的第几天
     */
    // 1971年1月1日是周五
    public static String dayOfTheWeek(int day, int month, int year) {
        String[] ans = new String[]{"Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
        int[] daysOfMonth1 = new int[]{31,28,31,30,31,30,31,31,30,31,30,31};
        int[] daysOfMonth2 = new int[]{31,29,31,30,31,30,31,31,30,31,30,31};
        int sum = 4;
        for(int i = 1971;i<year;i++){
            sum += 365;
            if(i%4==0 && i!=2100){
                sum++;
            }
        }
        if(year%4==0 && year!=2100){
            for(int i = 1;i<month;i++){
                sum += daysOfMonth2[i-1];
            }
        }else{
            for(int i = 1;i<month;i++){
                sum += daysOfMonth1[i-1];
            }
        }
        sum += day;
        return ans[sum%7];

    }

    /** ##DFS
     * 463. 岛屿的周长
     */
    static int[] dx = {0, 1, 0, -1};
    static int[] dy = {1, 0, -1, 0};

    public int islandPerimeter(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == 1) {
                    ans += dfs(i, j, grid, n, m);
                }
            }
        }
        return ans;
    }
    public int dfs(int x, int y, int[][] grid, int n, int m) {
        if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] == 0) {
            return 1;
        }
        if (grid[x][y] == 2) {
            return 0;
        }
        grid[x][y] = 2;
        int res = 0;
        for (int i = 0; i < 4; ++i) {
            int tx = x + dx[i];
            int ty = y + dy[i];
            res += dfs(tx, ty, grid, n, m);
        }
        return res;
    }

    /** ##
     * 665. 非递减数列
     */
    public static boolean checkPossibility(int[] nums) {
        if(nums.length < 3){
            return true;
        }
        int count = 0;
        for(int i=0;i<nums.length-1;i++){
            if(nums[i] > nums[i+1]){
                count++;
                if(count > 1){
                    break;
                }
                if(i-1 >=0&&nums[i-1] > nums[i+1]){
                    nums[i+1] = nums[i];
                }else{
                    nums[i] = nums[i+1];
                }
            }
        }
        return count <= 1;
    }

    /**
     * 283. 移动零
     */
    public static int[] moveZeroes(int[] nums) {
        int i = 0;
        for(int j=0;j<nums.length;j++){
            if(nums[j]!=0){
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                i++;
            }
        }
        return nums;
    }

    /**
     * 884. 两句话中的不常见单词
     */
    public String[] uncommonFromSentences(String A, String B) {
        String[] tmpA = A.split( " ");
        String[] tmpB = B.split(" ");
        Map<String,Integer> ans = new HashMap<>();
        for (String s : tmpA) {
            if (ans.containsKey(s)) {
                ans.put(s, ans.get(s) + 1);
            } else {
                ans.put(s, 1);
            }
        }
        for (String s : tmpB) {
            if (ans.containsKey(s)) {
                ans.put(s, ans.get(s) + 1);
            } else {
                ans.put(s, 1);
            }
        }
        List<String> res = new ArrayList<>();
        for(Map.Entry<String,Integer> entry:ans.entrySet()){
            if(entry.getValue()==1){
                res.add(entry.getKey());
            }
        }
        //List转数组
        return res.toArray(new String[res.size()]);
    }

    /** ##
     * 1030. 距离顺序排列矩阵单元格
     */
    public static int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        int[][] res = new int[R * C][2];
        res[0][0] = r0;
        res[0][1] = c0;
        int[] dr = {1, 1, -1, -1};
        int[] dc = {1, -1, -1, 1};
        int row = r0;
        int col = c0;
        int cnt = 1;
        while (cnt < R * C) {
            row--;
            for (int i = 0; i < 4; i++) {
                while ((i % 2 == 0 && row != r0) || (i % 2 != 0 && col != c0)) {
                    if (row >= 0 && row < R && col >= 0 && col < C) {
                        res[cnt][0] = row;
                        res[cnt][1] = col;
                        cnt++;
                    }
                    row += dr[i];
                    col += dc[i];
                }
            }
        }
        return res;
    }
    /**
     * 412. Fizz Buzz
     */
    public List<String> fizzBuzz(int n) {
        List<String> res = new ArrayList<>();
        for(int i = 1;i<=n;i++){
            if(i%3==0 && i%5==0){
                res.add("FizzBuzz");
            }else if(i%3==0){
                res.add("Fizz");
            }else if(i%5==0){
                res.add("Buzz");
            }else {
                res.add(String.valueOf(i));
            }
        }
        return res;
    }

    /**
     * 682. 棒球比赛
     */
    public static int calPoints(String[] ops) {
        int points = 0;
        List<Integer> tmp = new ArrayList<>();
        for (String op : ops) {
            switch (op) {
                case "C":
                    tmp.remove(tmp.size() - 1);
                    break;
                case "D":
                    tmp.add(tmp.get(tmp.size() - 1) * 2);
                    break;
                case "+":
                    tmp.add(tmp.get(tmp.size() - 1) + tmp.get(tmp.size() - 2));
                    break;
                default:
                    tmp.add(Integer.valueOf(op));
                    break;
            }
        }
        for(Integer number :tmp){
            points += number;
        }
        return points;
    }

    /**
     * 1518. 换酒问题
     */
    public int numWaterBottles(int numBottles, int numExchange) {
        int count = numBottles;
        while(numBottles>=numExchange){
            count += numBottles/numExchange;
            numBottles = numBottles%numExchange+numBottles/numExchange;
        }
        return count;
    }

    /**
     * 806. 写字符串需要的行数
     */
    public int[] numberOfLines(int[] widths, String S) {
        int row = 1;
        int col = 0;
        for(char c:S.toCharArray()){
            if((col+widths[c-'a'])>100){
                row++;
                col = widths[c-'a'];
            }else if((col+widths[c-'a'])==100){
                row++;
                col = 0;
            }else{
                col += widths[c-'a'];
            }
        }
        return new int[]{row,col};
    }

    /**
     * 509. 斐波那契数
     */
    public int fib(int N) {
        if(N<=1){
            return N;
        }
        int x1 = 0,x2 = 1,tmp;
        for(int i = 2;i<=N;i++){
            tmp = x2;
            x2 = x1+x2;
            x1 = tmp;
        }
        return x2;
    }

    /**
     * 566. 重塑矩阵
     */
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int row = nums.length;
        int col = nums[0].length;
        if (r * c == row * col) {
            int[][] res = new int[r][c];
            int rowIndex = 0, colIndex = 0;
            for (int[] num : nums) {
                for (int j = 0; j < col; j++) {
                    res[rowIndex][colIndex] = num[j];
                    colIndex++;
                    if (colIndex == c) {
                        rowIndex++;
                        colIndex = 0;
                    }
                }
            }
            return res;
        } else {
            return nums;
        }
    }

    /**
     * 1047. 删除字符串中的所有相邻重复项
     */
    public static String removeDuplicates(String S) {
        StringBuilder sb = new StringBuilder();
        int sbLength = 0;
        for(char c: S.toCharArray()){
            if(sbLength!=0 && c==sb.charAt(sbLength-1)){
                sb.deleteCharAt(sbLength-1);
                sbLength--;
            }else {
                sb.append(c);
                sbLength++;
            }
        }
        return sb.toString();
    }

    /**
     * 1550. 存在连续三个奇数的数组
     */
    public boolean threeConsecutiveOdds(int[] arr) {
        int count = 0;
        for(int num:arr){
            if(num%2==1){
                count++;
            }else{
                count = 0;
            }
            if(count==3){
                return true;
            }
        }
        return false;
    }

    /**
     * 1207. 独一无二的出现次数
     */
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer,Integer> tmp = new HashMap<>();
        for(int number:arr){
            if(tmp.containsKey(number)){
                tmp.put(number,tmp.get(number)+1);
            }else {
                tmp.put(number,1);
            }
        }
        Set<Integer> ans = new HashSet<>();
        for(Map.Entry<Integer,Integer> entry:tmp.entrySet()){
            if(ans.contains(entry.getValue())){
                return false;
            }else{
                ans.add(entry.getValue());
            }
        }
        return true;
    }

    /**
     * 999. 可以被一步捕获的棋子数
     */
    public static int numRookCaptures(char[][] board) {
        int row = board.length;
        int col = board[0].length;
        int count = 0;
        int index1 = 0,index2 = 0;
        for(int i = 0; i<row;i++){
            int j;
            for(j = 0; j<col;j++){
                if(board[i][j]=='R'){
                    index1 = i;
                    index2 = j;
                    break;
                }
            }
            if(j!=col){
                break;
            }
        }
        int tmp = index1;
        while(tmp>=0){
            if(board[tmp][index2]=='B'){
                break;
            }else if(board[tmp][index2]=='p'){
                count++;
                break;
            }
            tmp--;
        }
        tmp = index1;
        while(tmp<row){
            if(board[tmp][index2]=='B'){
                break;
            }else if(board[tmp][index2]=='p'){
                count++;
                break;
            }
            tmp++;
        }
        tmp = index2;
        while(tmp>=0){
            if(board[index1][tmp]=='B'){
                break;
            }else if(board[index1][tmp]=='p'){
                count++;
                break;
            }
            tmp--;
        }
        tmp = index2;
        while(tmp<col){
            if(board[index1][tmp]=='B'){
                break;
            }else if(board[index1][tmp]=='p'){
                count++;
                break;
            }
            tmp++;
        }
        return count;
    }

    /**
     * 922. 按奇偶排序数组 II
     */
    public int[] sortArrayByParityII(int[] A) {
        int[] res = new int[A.length];
        int index1 = 0;
        int index2 = 1;
        for (int value : A) {
            if (value % 2 == 0) {
                res[index1] = value;
                index1 += 2;
            } else {
                res[index2] = value;
                index2 += 2;
            }
        }
        return res;
    }
    public int[] sortArrayByParityII1(int[] A) {
        int j = 1;
        for(int i = 0;i<A.length;i+=2){
            if(A[i]%2==1){
                while(A[j]%2==1){
                    j += 2;
                }
                int tmp = A[i];
                A[i] = A[j];
                A[j] = tmp;
            }
        }
        return A;
    }

    /**
     * 476. 数字的补数
     */
    public static int findComplement(int num) {
        if(num==0) return 1;
        int temp = num;
        num = (num>>1) | num;
        num = (num>>2)| num;
        num = (num>>4)| num;
        num = (num>>8)| num;
        num = (num>>16)| num;
        return num & (-temp-1);
    }

    /**
     * 1475. 商品折扣后的最终价格
     */
    public static int[] finalPrices(int[] prices) {
        int len = prices.length;
        Stack<Integer> stack=new Stack<>();
        for(int i = 0; i < len; i++) {
            while(!stack.isEmpty() && prices[stack.peek()] >= prices[i]) {
                int index = stack.pop();    // java 的pop可以直接获取顶元素就不用像c++ 一样先top再pop了
                prices[index] -= prices[i];
            }
            stack.push(i);
        }
        return prices;
    }

    /**
     * 104. 二叉树的最大深度
     */
    public int maxDepth(TreeNode root) {
        int depth = 0;
        if(root==null){
            return depth;
        }
        // 要把root节点这个深度加上
        depth++;
        depth += Math.max(maxDepth(root.left),maxDepth(root.right));
        return depth;
    }

    /**
     * 206. 反转链表
     */
    public ListNode reverseList1(ListNode head) {
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null){
            ListNode temp = cur.next;
            cur.next = prev;
            prev = cur;
            cur = temp;
        }
        return prev;
    }
    public ListNode reverseList2(ListNode head) {
        if(head==null){
            return null;
        }
        List<Integer> tmp = new ArrayList<>();
        ListNode node = head;
        while(node!=null){
            tmp.add(node.val);
            node = node.next;
        }
        ListNode ans = new ListNode(tmp.get(tmp.size()-1));
        ListNode res = ans;
        for(int i = tmp.size()-2;i>=0;i--){
            res.next = new ListNode(tmp.get(i));
            res = res.next;
        }
        return ans;
    }

    /**
     * 144. 二叉树的前序遍历
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        Deque<TreeNode> temp = new LinkedList<>();
        TreeNode node = root;
        while(!temp.isEmpty() || node !=null){
            while(node!=null){
                res.add(node.val);
                temp.push(node);
                node = node.left;
            }
            node = temp.pop();
            node = node.right;
        }
        return res;
    }

    /**
     * 905. 按奇偶排序数组
     */
    //双指针
    public int[] sortArrayByParity(int[] A) {
        int i = 0, j = A.length - 1;
        while (i < j) {
            if (A[i]%2 > A[j]%2) {
                int tmp = A[i];
                A[i] = A[j];
                A[j] = tmp;
            }

            if (A[i] % 2 == 0) i++;
            if (A[j] % 2 == 1) j--;
        }

        return A;
    }

    /**
     * 1460. 通过翻转子数组使两个数组相等
     */
    public boolean canBeEqual(int[] target, int[] arr) {
        Arrays.sort(target);
        Arrays.sort(arr);
        for(int i = 0;i<arr.length;i++){
            if(target[i]!=arr[i]){
                return false;
            }
        }
        return true;
    }

    /**
     * 700. 二叉搜索树中的搜索
     */
    public TreeNode searchBST(TreeNode root, int val) {
        TreeNode temp = root;
        while(temp !=null){
            if(temp.val>val){
                temp = temp.left;
            }else if(temp.val<val){
                temp = temp.right;
            }else{
                return temp;
            }
        }
        return null;
    }

    /**
     * 1309. 解码字母到整数映射
     */
    public static String freqAlphabets(String s) {
        char[] data = new char[]{'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
                'r','s','t','u','v','w','x','y','z'};
        int length = s.length()-1;
        StringBuilder res = new StringBuilder();
        while(length>=0){
            if(s.charAt(length)=='#'){
                int temp = s.charAt(length - 1) - '0' + (s.charAt(length - 2) - '0') *10;
                res.append(data[temp-1]);
                length -= 3;
            }else{
                res.append(data[s.charAt(length) - '0' -1]);
                length--;
            }
        }
        return res.reverse().toString();
    }

    /**
     * 728. 自除数
     */
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> res = new ArrayList<>();
        for(int i = left;i<=right;i++){
            if(divideByMyself(i)){
                res.add(i);
            }
        }
        return res;
    }
    public boolean divideByMyself(int x){
        Set<Integer> tmp = new HashSet<>();
        int temp = x;
        while (x!=0){
            tmp.add(x%10);
            x /= 10;
        }
        for(Integer num:tmp){
            if(num==0){
                return false;
            }
            if(temp%num!=0){
                return false;
            }
        }
        return true;
    }

    /**
     * 461. 汉明距离
     */
    public static int hammingDistance(int x, int y) {
        int xor = x ^ y;
        int distance = 0;
        while (xor != 0) {
            if (xor % 2 == 1)
                distance += 1;
            xor = xor >> 1;
        }
        return distance;
    }
    public int hammingDistance1(int x, int y) {
        int xor = x ^ y;
        int distance = 0;
        while (xor != 0) {
            distance += 1;
            // remove the rightmost bit of '1'
            xor = xor & (xor - 1);
        }
        return distance;
    }

    /**
     * 832. 翻转图像
     */
    public int[][] flipAndInvertImage(int[][] A) {
        int row = A.length;
        int col = A[0].length;
        for(int i = 0;i<row;i++){
            for(int j = 0;j<col;j++){
                A[i][j] ^= 1;
            }
            for(int k = 0;k<col/2;k++){
                int temp = A[i][k];
                A[i][k] = A[i][col-1-k];
                A[i][col-1-k] = temp;
            }
        }
        return A;
    }

    /**
     * 1351. 统计有序矩阵中的负数
     */
    public static int countNegatives(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int count = 0,tmprow = row;
        int right = col-1;
        for (int[] ints : grid) {
            while (right >= 0 && ints[right] < 0) {
                right -= 1;
            }
            count += (col - 1 - right) * tmprow;
            col = right + 1;
            tmprow--;
        }
        return count;
    }

    /**
     * 804. 唯一摩尔斯密码词
     */
    public int uniqueMorseRepresentations(String[] words) {
        String[] code = new String[]{".-","-...","-.-.","-..",".","..-.","--.","....",
                "..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-",
                "..-","...-",".--","-..-","-.--","--.."};
        Set<String> res = new HashSet<>();
        for(String word : words){
            StringBuffer tmp = new StringBuffer();
            for(int i = 0;i<word.length();i++){
                tmp.append(code[word.charAt(i)-'a']);
            }

            res.add(tmp.toString());
        }
        return res.size();
    }
    /**
     * 709. 转换成小写字母
     */
    public String toLowerCase(String str) {
        int len = str.length();
        StringBuffer res = new StringBuffer();
        for(int i = 0;i<len;i++){
            if((int)str.charAt(i)<=90 && (int)str.charAt(i)>=65){
                res.append((char)((int)str.charAt(i)+32));
            }else{
                res.append(str.charAt(i));
            }
        }
        return res.toString();
    }

    /**
     *
     */
    public int maxProduct(int[] nums) {
        Arrays.sort(nums);
        return Math.max((nums[nums.length - 1] - 1) * (nums[nums.length - 2] - 1), (nums[0] - 1) * (nums[1] - 1));

    }

    /**
     * 1528. 重新排列字符串
     */
    public String restoreString(String s, int[] indices) {
        int len = indices.length;
        char[] res = new char[len];
        for(int i = len-1; i>=0;i--){
            res[indices[i]] = s.charAt(i);
        }
        return new String(res);
    }

    /**
     * 1365. 有多少小于当前数字的数字
     */
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        for(int i = 0;i<len;i++){
            int count = 0;
            for(int j = 0;j<len;j++){
                if(i==j){
                    continue;
                }
                if(nums[i]>nums[j]){
                    count++;
                }
            }
            res[i] = count;
        }
        return res;
    }
    /**
     * 1342. 将数字变成 0 的操作次数
     */
    public int numberOfSteps (int num) {
        int count = Integer.bitCount(num) - 1;
        num |= num >> 1;
        num |= num >> 2;
        num |= num >> 4;
        num |= num >> 8;
        num |= num >> 16;
        return count + Integer.bitCount(num);
    }

    /**
     * 	1313.解压缩编码列表
     */
    public int[] decompressRLElist(int[] nums) {
        int count = 0;
        for(int i = 0;i<nums.length;i++){
            count += nums[i];
            i++;
        }
        int[] res = new int[count];
        int index = 0;
        int tmp = 0;
        while(index<nums.length-1){
            for(int i = 0;i<nums[index];i++){
                res[tmp+i] = nums[index+1];
            }
            tmp += nums[index];
            index += 2;
        }
        return res;
    }

    /**
     * 1108. IP 地址无效化
     */
    public String defangIPaddr(String address) {
        StringBuilder str = new StringBuilder();
        for(int i = 0; i<address.length();i++){
            if(address.charAt(i)=='.'){
                str.append("[.]");
            }else{
                str.append(address.charAt(i));
            }
        }
        return str.toString();
    }

    /**
     * 1512. 好数对的数目
     */
    public int numIdenticalPairs(int[] nums) {
        Map<Integer,Integer> numMap = new HashMap<>();
        for(int i = 0;i<nums.length;i++){
            if(numMap.containsKey(nums[i])){
                numMap.put(nums[i],numMap.get(nums[i])+1);
            }else {
                numMap.put(nums[i],1);
            }
        }
        int res = 0;
        for(Map.Entry<Integer,Integer> entry:numMap.entrySet()){
            res +=entry.getValue()*(entry.getValue()-1)/2;
        }
        return res;
    }

    /**
     * 1480. 一维数组的动态和
     */
    public int[] runningSum(int[] nums) {
        int[] res = new int[nums.length];
        int tmp = 0;
        for(int i = 0;i<nums.length;i++){
            tmp += nums[i];
            res[i] = tmp;
        }
        return res;
    }

    /**
     * 1323. 6 和 9 组成的最大数字
     */
    public int maximum69Number (int num) {
        return Integer.parseInt(String.valueOf(num).replaceFirst("6","9"));
    }

    /**
     * 1299. 将每个元素替换为右侧最大元素
     */
    public int[] replaceElements(int[] arr) {
        int[] res = new int[arr.length];
        int tmp = Integer.MIN_VALUE,len = arr.length;
        res[len-1] = -1;
        for(int i = len-1;i>=1;i--){
            if(arr[i]>tmp){
                tmp = arr[i];
            }
            res[i-1] = tmp;
        }
        return res;
    }

    /**
     * 1295. 统计位数为偶数的数字
     */
    public int findNumbers(int[] nums) {
        int ans = 0;
        for(int num:nums){
            if(count(num)%2==0){
                ans++;
            }
        }
        return ans;
    }
    public int count(int num){
        int ans = 0;
        while(num!=0){
            num /= 10;
            ans++;
        }
        return ans;
    }

    /**
     * 1290. 二进制链表转整数
     */
    public int getDecimalValue(ListNode head) {
        int res = 0;
        ListNode tmp = head;
        while (tmp != null){
            res = res*2+tmp.val;
            tmp = tmp.next;
        }

        return res;
    }

    /**
     * 234. 回文链表
     */
    //O(n),O(n)
    public boolean isPalindrome(ListNode head) {
        ListNode tmp = head;
        List<Integer> nums = new ArrayList<>();
        while(tmp != null){
            nums.add(tmp.val);
            tmp = tmp.next;
        }
        int left = 0,right = nums.size()-1;
        while(left < right){
            if(!nums.get(left).equals(nums.get(right))){
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    //快慢指针，骚得很
    public boolean isPalindrome1(ListNode head) {
        if (head == null) {
            return true;
        }
        // 找到前半部分链表的尾节点并反转后半部分链表
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);
        // 判断是否回文
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) {
                result = false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        // 还原链表并返回结果
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }
    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }
    //双指针，一个步长2，一个步长1，当快指针到链表尾部时，慢指针正好在中间
    private ListNode endOfFirstHalf(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     *704. 二分查找
     */
    public static int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        int tmp;
        while (left!=right && left !=(right-1)){
            tmp = (left+right)/2;
            if(nums[tmp]==target){
                return tmp;
            }else if(nums[tmp]>target){
                right = tmp;
            }else{
                left = tmp;
            }
        }
        if(nums[left]==target){
            return left;
        }else if(nums[right]==target){
            return right;
        }
        return -1;
    }

    /**
     *690. 员工的重要性
     */
    Map<Integer, Employee> employeeMap;
    public int getImportance(List<Employee> employees, int id) {
        employeeMap = new HashMap();
        for (Employee e: employees) employeeMap.put(e.id, e);
        return dfs(id);
    }
    public int dfs(int id) {
        Employee employee = employeeMap.get(id);
        int ans = employee.importance;
        for (Integer subid: employee.subordinates)
            ans += dfs(subid);
        return ans;
    }

    /**
     *661. 图片平滑器
     */
    public static int[][] imageSmoother(int[][] M) {
        int row = M.length;
        int col =M[0].length;
        int[][] ans = new int[row][col];
        for(int i = 0;i<row;i++){
            for(int j = 0;j<col;j++){
                ans[i][j] = forAverage(i,j,row,col,M);
            }
        }
        return ans;
    }
    public static int forAverage(int n,int m,int row,int col,int[][] M){
        int[] tmp = new int[]{-1,0,1};
        int sum = 0;
        int count = 0;
        for(int i = 0;i<3;i++){
            for(int j = 0;j<3;j++){
                if((n+tmp[i])>=0 && (n+tmp[i])<row && (m+tmp[j])>=0 && (m+tmp[j])<col){
                    sum +=M[n+tmp[i]][m+tmp[j]];
                    count++;
                }
            }
        }
        return (int) Math.floor(sum/count);

    }

    /**
     *657. 机器人能否返回原点
     */
    public boolean judgeCircle(String moves) {
        int len = moves.length();
        int sum1 = 0,sum2 = 0;
        for(int i = 0; i<len;i++){
            sum1 +=judgeDirection1(moves.charAt(i));
            sum2 +=judgeDirection2(moves.charAt(i));
        }
        return sum1==0 && sum2==0;
    }
    public int judgeDirection1(char s){
        switch (s){
            case 'R':
                return 1;
            case 'L':
                return -1;
        }
        return 0;
    }
    public int judgeDirection2(char s){
        switch (s){
            case 'U':
                return 1;
            case 'D':
                return -1;
        }
        return 0;
    }

    /**
     * 1287. 有序数组中出现次数超过25%的元素
     */
    //用Map对arr中的数字进行计数
    public int findSpecialInteger(int[] arr) {
        Map<Integer,Integer> tmp = new HashMap<>();
        for (int value : arr) {
            if (tmp.containsKey(value)) {
                tmp.put(value, tmp.get(value) + 1);
            } else {
                tmp.put(value, 1);
            }
        }
        int res = 0;
        for(Map.Entry<Integer,Integer> entity:tmp.entrySet()){
            if(entity.getValue() > (arr.length*0.25)){
                res = entity.getKey();
                break;
            }
        }
        return res;
    }
    //利用数组长度的1/4
    public int findSpecialInteger1(int[] arr) {
        int len = arr.length/4;
        for(int i = 0;i<arr.length;i++){
            if(arr[i] == arr[i+len]){
                return arr[i];
            }
        }
        return 0;
    }

    /**
     * 1486. 数组异或操作
     */
    public int xorOperation(int n, int start) {
        int res = 0;
        for(int i = 0; i < n;i++){
            res ^= start+2*i;
        }
        return res;
    }

    /**
     * 1502. 判断能否形成等差数列
     */
    public boolean canMakeArithmeticProgression(int[] arr) {
        Arrays.sort(arr);
        Set<Integer> diffSet = new HashSet<>();
        diffSet.add(arr[1]-arr[0]);
        for(int i = 2;i<arr.length;i++){
            if(!diffSet.contains(arr[i]-arr[i-1])){
                return false;
            }
        }
        return true;
    }

    /**
     * 1450. 在既定时间做作业的学生人数
     */
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        int count = 0;
        int len = startTime.length;
        for(int i = 0;i<len;i++){
            if(startTime[i]<=queryTime && endTime[i]>=queryTime){
                count++;
            }
        }
        return count;
    }

    /**
     * 867. 转置矩阵
     */
    public int[][] transpose(int[][] A) {
        int row = A.length;
        int col = A[0].length;
        int[][] ans = new int[col][row];
        for(int i = 0;i<row;i++){
            for(int j = 0;j<col;j++){
                ans[j][i] = A[i][j];
            }
        }
        return ans;
    }

    /**
     * 1002. 查找常用字符
     */
    public List<String> commonChars(String[] A) {
        int[] minFreq = new int[26];
        Arrays.fill(minFreq,Integer.MAX_VALUE);
        for(String word :A){
            int[] freq = new int[26];
            int length = word.length();
            for(int i =0;i<length;i++){
                freq[word.charAt(i)-'a']++;
            }
            for(int j = 0;j<26;j++){
                minFreq[j] = Math.min(minFreq[j],freq[j]);
            }
        }
        List<String> ans  = new ArrayList<>();
        for(int i = 0;i < 26;i++){
            for(int j = 0;j<minFreq[i];j++){
                ans.add(String.valueOf((char)(i+'a')));
            }
        }
        return ans;
    }

    /**
     * 136. 只出现一次的数字
     */
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int num:nums){
            res ^= num;
        }
        return res;
    }

    /**
     * 763. 划分字母区间
     */
    public static List<Integer> partitionLabels(String S) {
        int[] last = new int[26];
        int len = S.length();
        for(int i = 0; i<len;i++){
            last[S.charAt(i)-'a'] = i;
        }
        List<Integer> partition = new ArrayList<>();
        int start = 0,end = 0;
        for(int i = 0;i<len;i++){
            end = Math.max(end,last[S.charAt(i)-'a']);
            if(i==end){
                partition.add(end-start+1);
                start = end+1;
            }
        }
        return partition;
    }

    /**
     * 202. 快乐数
     */
    public boolean isHappy(int n) {
        Set<Integer> tmpSet = new HashSet<>();
        while(n!=1 && !tmpSet.contains(n)){
            tmpSet.add(n);
            n = getNext(n);
        }
        return n==1;
    }
    public int getNext(int n){
        int res = 0;
        while(n!=0){
            res += (n%10)*(n%10);
            n /= 10;
        }
        return res;
    }

    /** ##
     * 52. N皇后 II
     */
    public int totalNQueens(int n) {
        Set<Integer> columns = new HashSet<>();
        Set<Integer> diagonals1 = new HashSet<>();
        Set<Integer> diagonals2 = new HashSet<>();
        return backtrack(n, 0, columns, diagonals1, diagonals2);
    }
    public int backtrack(int n, int row, Set<Integer> columns, Set<Integer> diagonals1, Set<Integer> diagonals2) {
        if (row == n) {
            return 1;
        } else {
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (columns.contains(i)) {
                    continue;
                }
                int diagonal1 = row - i;
                if (diagonals1.contains(diagonal1)) {
                    continue;
                }
                int diagonal2 = row + i;
                if (diagonals2.contains(diagonal2)) {
                    continue;
                }
                columns.add(i);
                diagonals1.add(diagonal1);
                diagonals2.add(diagonal2);
                count += backtrack(n, row + 1, columns, diagonals1, diagonals2);
                columns.remove(i);
                diagonals1.remove(diagonal1);
                diagonals2.remove(diagonal2);
            }
            return count;
        }
    }

    /**
     * 19. 删除链表的倒数第N个节点
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode temp = new ListNode(0,head);
        int len = 0;
        ListNode tmp = head;
        while(tmp != null){
            len++;
            tmp = tmp.next;
        }
        ListNode cur = temp;
        for(int i = 1; i<len-n+1;i++){
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return temp.next;
    }

    /**
     * 925. 长按键入
     */
    public boolean isLongPressedName(String name, String typed) {
        int nameLen = name.length();
        int typedLen = typed.length();
        int i = 0;
        int j = 0;
        while( j < typedLen){
            if(i < nameLen && name.charAt(i) == typed.charAt(j)){
                i++;
                j++;
            }else if(j > 0 && typed.charAt(j-1) == typed.charAt(j)){
                j++;
            }else{
                return false;
            }
        }
        return i == nameLen;
    }

    /**
     * 1314. 矩阵区域和
     */
    public int[][] matrixBlockSum(int[][] mat, int K) {
        int rowLen = mat.length;
        int coLen = mat[0].length;
        int[][] ans = new int[rowLen][coLen];
        for(int i = 0;i<rowLen;i++){
            for(int j = 0;j<coLen;j++){
                int temp = 0;
                for(int n = Math.max(0,i-K);n<=Math.min(rowLen-1,i+K);n++){
                    for(int m = Math.max(0,j-K);m<=Math.min(coLen-1,j+K);m++){
                        temp += mat[n][m];
                    }
                }
                ans[i][j] = temp;
            }
        }
        return ans;
    }

    /**
     * 171. Excel表列序号
     */
    public int titleToNumber(String s) {
        int len = s.length();
        int sum = 0;
        for(int i = len-1;i>=0;i--){
            sum += (s.charAt(i)-'A')*Math.pow(26,len-1-i);
        }
        return sum;
    }
    public int ZifuToShuZi(char x){
        switch (x){
            case 'A':
                return 1;
            case 'B':
                return 2;
            case 'C':
                return 3;
            case 'D':
                return 4;
            case 'E':
                return 5;
            case 'F':
                return 6;
            case 'G':
                return 7;
            case 'H':
                return 8;
            case 'I':
                return 9;
            case 'J':
                return 10;
            case 'K':
                return 11;
            case 'L':
                return 12;
            case 'M':
                return 13;
            case 'N':
                return 14;
            case 'O':
                return 15;
            case 'P':
                return 16;
            case 'Q':
                return 17;
            case 'R':
                return 18;
            case 'S':
                return 19;
            case 'T':
                return 20;
            case 'U':
                return 21;
            case 'V':
                return 22;
            case 'W':
                return 23;
            case 'X':
                return 24;
            case 'Y':
                return 25;
            case 'Z':
                return 26;
            default:
                return 0;
        }
    }

    /**
     * 122. 买卖股票的最佳时机 II
     */
    public int maxProfit(int[] prices) {
        int maxprofit = 0;
        for(int i = 1; i<prices.length;i++){
            if(prices[i]>prices[i-1]){
                maxprofit += prices[i]-prices[i-1];
            }
        }
        return maxprofit;
    }

    /**
     * 191. 位1的个数
     */
    public int hammingWeight(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

    /**
     * 143. 重排链表
     */
    public void reorderList(ListNode head) {
        if(head == null){
            return;
        }
        List<ListNode> tmp =new ArrayList<>();
        ListNode node = head;
        while(node!=null){
            tmp.add(node);
            node = node.next;
        }
        int j = tmp.size()-1;
        int i = 0;
        while(i < j){
            tmp.get(i).next = tmp.get(j);
            i++;
            if(i==j){
                break;
            }
            tmp.get(j).next = tmp.get(i);
            j--;
        }
        tmp.get(i).next = null;
    }

    /**
     * 67. 二进制求和
     */
    public String addBinary(String a, String b) {
        int count = 0;
        int indexA = a.length()-1;
        int indexB = b.length()-1;
        StringBuffer res = new StringBuffer();
        while(indexA>=0 || indexB>=0){
            int n1 = indexA==-1? 0:Integer.parseInt(String.valueOf(a.charAt(indexA)));
            int n2 = indexB==-1? 0:Integer.parseInt(String.valueOf(b.charAt(indexB)));
            if(indexA!=-1){
                indexA--;
            }
            if(indexB!=-1){
                indexB--;
            }
            int tmp = n1+n2+count;
            count = tmp/2;
            res.append(tmp%2);
        }
        if(count!=0){
            res.append(count);
        }
        res.reverse();
        return res.toString();
    }

    /**
     * 118. 杨辉三角
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if(numRows<=0){
            return res;
        }
        res.add(new ArrayList<>());
        res.get(0).add(1);
        for(int i = 1; i<numRows;i++){
            List<Integer> row = new ArrayList<>();
            List<Integer> preRow = res.get(i-1);
            row.add(1);
            for(int j = 1;j<i;j++){
                row.add(preRow.get(j-1)+preRow.get(j));
            }
            row.add(1);
            res.add(row);
        }
        return res;

    }

    /**
     * 844. 比较含退格的字符串
     */
    public boolean backspaceCompare(String S, String T) {
        return build(S).equals(build(T));
    }
    public String build(String tmp){
        StringBuffer res = new StringBuffer();
        int len = tmp.length();
        for(int i = 0;i<len;i++){
            char ch = tmp.charAt(i);
            if(ch != '#'){
                res.append(ch);
            }else{
                if(res.length()>0){
                    res.deleteCharAt(res.length()-1);
                }
            }
        }
        return res.toString();
    }

    /**
     *
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            tail.next = new ListNode(carry);
        }
        return head;
    }

    /**
     *
     */
    public int[] sortedSquares(int[] A) {
        int n = A.length;
        int[] res = new int[n];
        for(int i = 0,j = n-1,pos = n-1;i<=j;){
            if(A[i]*A[i]>A[j]*A[j]){
                res[pos] = A[i]*A[i];
                i++;
            }else{
                res[pos] = A[j]*A[j];
                j--;
            }
            pos--;
        }
        return res;
    }

    /** %%%%%
     *
     */
    public boolean checkPalindromeFormation(String a, String b) {
        char[] strA = a.toCharArray();
        char[] strB = b.toCharArray();
        int left = 0;
        int len = strA.length;
        while(left<=len/2){
            if(strA[left] == strB[len-1-left]){
                left++;
            }else {
                break;
            }
        }
        if(left>=len/2){
            return true;
        }
        return isPa(a.substring(left, len - left)) || isPa(b.substring(left, len - left));
    }
    private boolean isPa(String a){
        char[] str = a.toCharArray();
        int length = str.length;
        for(int i = 0; i<length/2;i++){
            if(str[i]!=str[length-1-i]){
                return false;
            }
        }
        return true;
    }

    /**
     * 1614. 括号的最大嵌套深度
     */
    public int maxDepth(String s) {
        if(!s.contains("(")){
            return 0;
        }
        int len = s.length();
        int count = 0;
        int res = 0;
        for(int j = 0; j < len; j++){
            if(s.charAt(j) == '('){
                count++;
                if(count > res){
                    res = count;
                }
            }else if(s.charAt(j)==')'){
                count--;
            }
        }
        return res;
    }

    /**
     * 66. 加一
     */
    public int[] plusOne(int[] digits) {
        int length = digits.length;
        int[] res = new int[length + 1];
        digits[length - 1] += 1;
        int count = 0;
        if(digits[length - 1] == 10){
            count = 1;
            digits[length - 1] = 0;
        }
        for(int i = length - 2; i >= 0; i--) {
            res[i] = digits[i] + count;
            digits[i] += count;
            if(res[i] < 10) {
                count = 0;
            } else {
                res[i] = 0;
                digits[i] = 0;
                count = 1;
            }
        }
        if(count == 1) {
            res[0] = 1;
            return res;
        }
        return digits;
    }

    /**
     * 判断是不是回文数
     */
    public boolean isPalindrome(int x) {
        if(x < 0 || (x%10 == 0 && x != 0)) return false;
        if(x < 10) return true;
        int temp = x;
        int reverseNumber = 0;
        while(x >= 10){
            reverseNumber = reverseNumber * 10 + x % 10;
            x /= 10;
        }
        return temp == reverseNumber;
    }

    /**
     * 归并排序
     */
    public void merge(int[] A, int m, int[] B, int n) {
        int i = 0 ,j = 0;
        int[] temp = new int[m + n];
        int tail = 0;
        while(tail < m + n && i < m && j < n){
            if(A[i]<=B[j]){
                temp[tail] = A[i++];
            }
            if(A[i]>B[j]){
                temp[tail] = B[j++];
            }
            tail++;
        }
        if(i == m-1){
            while(tail < n+m){
                temp[tail++] = B[j++];
            }
        }else{
            while(tail<n+m){
                temp[tail++] = A[i++];
            }
        }
        A = temp;
    }
}

