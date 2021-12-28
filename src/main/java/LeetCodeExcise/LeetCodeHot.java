package LeetCodeExcise;

import java.util.*;

public class LeetCodeHot {
    public static void main(String[] args) {
        String x = "   -42";
    }

    /** ##
     * 8. 字符串转换整数 (atoi)
     */
    public int myAtoi(String str) {
        Automaton automaton = new Automaton();
        int length = str.length();
        for (int i = 0; i < length; ++i) {
            automaton.get(str.charAt(i));
        }
        return (int) (automaton.sign * automaton.ans);
    }

    /**
     * 739. 每日温度
     */
    public static int[] dailyTemperatures(int[] T) {
        int length = T.length;
        int[] ans = new int[length];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < length; i++) {
            int temperature = T[i];
            while (!stack.isEmpty() && temperature > T[stack.peek()]) {
                int prevIndex = stack.pop();
                ans[prevIndex] = i - prevIndex;
            }
            stack.push(i);
        }
        return ans;
    }
    public static int[] dailyTemperatures1(int[] T) {
        int length = T.length;
        for(int i = 0; i < length; i++) {
            int count = 0;
            for(int j = i + 1; j < length; j++) {
                count++;
                if(T[j] > T[i]) {
                    break;
                }
            }
            if(count == length - 1 - i && T[i] >= T[length - 1]) count = 0;
            T[i] = count;
        }
        return T;
    }

    /**
     * 48. 旋转图像
     * 先转置，后左右翻转
     */
    public static void rotate(int[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        int tmp = 0;
        for(int i = 0; i < row; i++) {
            for(int j = i; j < col; j++) {
                tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col/2; j++) {
                tmp = matrix[i][j];
                matrix[i][j] = matrix[i][col - 1 - j];
                matrix[i][col - 1 - j] = tmp;
            }
        }
    }

    /**
     * 287. 寻找重复数
     */
    public int findDuplicate(int[] nums) {
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != (i+1)) {
                return nums[i];
            }
        }
        return 0;
    }

    /**
     * 221. 最大正方形
     */
    public int maximalSquare(char[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        int[] height = new int[col], left = new int[col], right = new int[col];
        Arrays.fill(right,col);
        int res = 0;
        for(int i = 0; i < row; i++) {
            int curLeft = 0, curRight = col;
            for(int j = 0; j < col; j++) {
                if(matrix[i][j] == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j],curLeft);
                } else {
                    height[j] = 0;
                    curLeft = j + 1;
                    left[j] = 0;
                }
            }
            for(int j = col - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j],curRight);
                } else {
                    curRight = j;
                    right[j] = col;
                }
            }
            for(int j = 0; j < col; j++) {
                res = (int) Math.max(res,Math.pow(Math.min(right[j] - left[j],height[j]),2));
            }
        }
        return res;
    }

    /**
     * 94. 二叉树的中序遍历
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        if(root.left != null) {
            res.addAll(inorderTraversal(root.left));
        }
        res.add(root.val);
        if(root.right != null) {
            res.addAll(inorderTraversal(root.right));
        }
        return res;
    }

    /**
     * 85. 最大矩形
     */
    public int maximalRectangle(char[][] matrix) {
        if(matrix.length == 0) return 0;
        int row = matrix.length, col = matrix[0].length;
        int[] left = new int[col], right = new int[col], height = new int[col];
        Arrays.fill(right,col);
        int res = 0;
        for(int i = 0; i < row; i++) {
            int curLeft = 0, curRight = col;
            for(int j = 0; j < col; j++) {
                if(matrix[i][j] == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j],curLeft);
                } else {
                    height[j] = 0;
                    left[j] = 0;
                    curLeft = j + 1;
                }
            }
            for(int j = col - 1; j >= 0; j--) {
                if(matrix[i][j] == '1') {
                    right[j] = Math.min(right[j],curRight);
                } else {
                    right[j] = col;
                    curRight = j;
                }
            }
            for(int j = 0; j < col; j++) {
                res = Math.max(res,(right[j] - left[j]) * height[j]);
            }
        }
        return res;
    }

    /**
     * 64. 最小路径和
     */
    public int minPathSum(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int[][] res = new int[row][col];
        res[0][0] = grid[0][0];
        for(int i = 1; i < col; i++) {
            res[0][i] = res[0][i-1] + grid[0][i];
        }
        for(int i = 1; i < row; i++) {
            res[i][0] = res[i-1][0] + grid[i][0];
        }
        for(int i = 1; i < row; i++) {
            for(int j = 1; j < col; j++) {
                res[i][j] = Math.min(res[i][j-1],res[i-1][j]) + grid[i][j];
            }
        }
        return res[row-1][col-1];
    }

    /** #
     * 42. 接雨水
     */
    public int trap(int[] height) {
        if(height == null || height.length <= 1) return 0;
        int length = height.length;
        int res = 0;
        int[] left_max = new int[length];
        left_max[0] = height[0];
        int[] right_max = new int[length];
        right_max[length-1] = height[length-1];
        for(int i = 1; i < length; i++) {
            left_max[i] = Math.max(height[i],left_max[i-1]);
        }
        for(int j = length - 2; j >= 0; j--) {
            right_max[j] = Math.max(height[j],right_max[j+1]);
        }
        for(int i = 1; i < length - 1; i++) {
            res += Math.max(left_max[i],right_max[i]) - height[i];
        }
        return res;
    }

    /** ##
     * 46. 全排列
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> output = new ArrayList<>();
        for (int num : nums) output.add(num);
        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }
    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) res.add(new ArrayList<>(output));
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }

    /**
     * 33. 搜索旋转排序数组
     */
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) return -1;
        if (n == 1) return nums[0] == target ? 0 : -1;
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid])  r = mid - 1;
                else l = mid + 1;
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) l = mid + 1;
                else r = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 56. 合并区间
     */
    public static int[][] merge(int[][] intervals) {
        Arrays.sort(intervals,(o1,o2) -> {
            if(o1[0] != o2[0]) return o1[0] - o2[0];
            return o1[1] - o2[1];
        });
        List<int[]> res = new ArrayList<>();
        int[] tmp = new int[]{intervals[0][0],intervals[0][1]};
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] > tmp[1] || intervals[i][1] < tmp[0]) {
                res.add(tmp);
                tmp = intervals[i];
            } else if(intervals[i][0] <= tmp[0] && intervals[i][1] >= tmp[0] && intervals[i][1] <= tmp[1]) {
                tmp[0] = intervals[i][0];
            } else if(intervals[i][0] >= tmp[0] && intervals[i][0] <= tmp[1] && intervals[i][1] >= tmp[1]) {
                tmp[1] = intervals[i][1];
            } else if(intervals[i][0] < tmp[0] && intervals[i][1] > tmp[1]) {
                tmp = intervals[i];
            }
        }
        res.add(tmp);
        return res.toArray(new int[res.size()][2]);
    }

    /** 双指针
     * 11. 盛最多水的容器
     */
    public int maxArea(int[] height) {
        int length = height.length;
        int res = Integer.MIN_VALUE;
        int left = 0, right = length - 1;
        while(left < right) {
            res = Math.max(res, Math.min(height[left], height[right]) * (right - left));
            if(height[left] >= height[right]) right--;
            else left++;
        }
        return res;
    }

    /** 快排手写
     * 215. 数组中的第K个最大元素
     */
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }
}
