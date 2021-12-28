package LeetCodeExcise;

import java.util.*;

public class LeetCodeWeeklySolution {
    public static void main(String[] args) {
        int[] n = new int[]{2,1,6,4};
        int res = waysToMakeFair(n);
        String s = "1001";
        boolean result = checkOnesSegment(s);
    }

    /**
     * 第 231 周赛
     */
    public static boolean checkOnesSegment(String s) {
        if(s.length() == 1) return true;
        List<Integer> res = new ArrayList<>();
        int count = 0;
        for(char ch : s.toCharArray()) {
            if(ch == '1') {
                count++;
            } else if(count >= 1) {
                res.add(count);
                count = 0;
            }
            if(count != 0 && res.size() > 0) {
                return false;
            }
        }
        if(count != 0) res.add(count);
        return res.size() == 1;
    }
    public int minElements(int[] nums, int limit, int goal) {
        long sum = 0;
        for(int x : nums) {
            sum += x;
        }
        long diff = goal - sum;
        long tmp = Math.abs(diff) % limit;
        return (int) (tmp == 0 ? Math.abs(diff) / limit : Math.abs(diff) / limit + 1);
    }
    /**
     * 第 224 周赛
     */
    public int countGoodRectangles(int[][] rectangles) {
        int count = 0;
        int maxLength = 0;
        for (int[] rectangle : rectangles) {
            maxLength = Math.max(maxLength, Math.min(rectangle[0], rectangle[1]));
        }
        for (int[] rectangle : rectangles) {
            if(maxLength == Math.min(rectangle[0], rectangle[1])) {
                count++;
            }
        }
        return count;
    }
    public int tupleSameProduct(int[] nums) {
        if(nums.length < 4) return 0;
        int count = 0;
        Map<Integer,Integer> res = new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            for(int j = i + 1; j < nums.length; j++) {
                int key = nums[i] * nums[j];
                res.put(key,res.getOrDefault(key,0) + 1);
            }
        }
        for(int key : res.keySet()) {
            count += res.get(key) > 1 ? res.get(key) * (res.get(key) - 1) * 4 : 0;
        }
        return count;
    }
    /**
     * 第 217 周赛
     */
    public int maximumWealth(int[][] accounts) {
        int row = accounts.length;
        int col = accounts[0].length;
        int res = Integer.MIN_VALUE;
        for(int i = 0; i < row; i++) {
            int tmp = 0;
            for(int j = 0; j < col; j++) {
                tmp += accounts[i][j];
            }
            res = Math.max(tmp,res);
        }
        return res;
    }
    /**
     * 第 216 周赛
     */
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuffer res1 = new StringBuffer();
        for(String str : word1) {
            res1.append(str);
        }
        StringBuffer res2 = new StringBuffer();
        for(String str : word2) {
            res2.append(str);
        }
        return res1.toString().equals(res2.toString());
    }
    public String getSmallestString(int n, int k) {
        int[] result = new int[n];
        Arrays.fill(result,k/n);
        int res = k - n * result[0];
        int index = 0;
        for(int i = n - 1; i >= 0; i--) {
            int tmp = 0;
            while(result[i] < 26 && index < i) {
                if(result[i] + res > 26){
                    tmp = result[i];
                    result[i] = 26;
                    res -= 26 - tmp;
                } else if(result[i] + res + result[index] - 1 > 26){
                    tmp = result[i];
                    result[i] = 26;
                    result[index] = result[index] - (26 - tmp - res);
                    res = 0;
                } else {
                    tmp = result[index];
                    result[index] = 1;
                    index++;
                    result[i] += tmp + res -1;
                    res = 0;
                }
            }
        }
        StringBuffer ans = new StringBuffer();
        for(int num : result) {
            ans.append((char)('a' + num -1));
        }
        return ans.toString();
    }

    /**
     * 1664. 生成平衡数组的方案数
     */
    public static int waysToMakeFair(int[] nums) {
        int length = nums.length;
        int[] dp = new int[length + 1];
        for(int i = 1; i < length + 1; i++) {
            dp[i] = dp[i - 1] + i%2 == 1 ? nums[i - 1] : (-1) * nums[i - 1];
        }
        int count = 0;
        for(int i = 1; i < length + 1; i++) {
            count += dp[i - 1] == (dp[length] - dp[i]) ? 1 : 0;
        }
        return count;
    }

    // 第 212 周赛
    /**[-12,-9,-3,-12,-6,15,20,-25,-20,-15,-10]
     [0,1,6,4,8,7]
     [4,4,9,7,9,10]
     * 1629. 按键持续时间最长的键
     */
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        char res = keysPressed.charAt(0);
        int time = releaseTimes[0];
        for(int i = 1; i < releaseTimes.length; i++) {
            if((releaseTimes[i] - releaseTimes[i-1]) > time) {
                time = releaseTimes[i] - releaseTimes[i-1];
                res = keysPressed.charAt(i);
            }else if((releaseTimes[i] - releaseTimes[i-1]) == time) {
                if(keysPressed.charAt(i) > res) {
                    res = keysPressed.charAt(i);
                }
            }
        }
        return res;
    }

    /**
     * 1630. 等差子数组
     * @param nums
     * @param l
     * @param r
     * @return
     */
    public static List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        List<Boolean> result = new ArrayList<>();
        for(int i = 0; i < l.length; i++ ) {
            result.add(checkArithmeticArray(nums,l[i],r[i]));
        }
        return result;
    }
    public static boolean checkArithmeticArray(int[] arr, int left, int right) {
        if(right - left < 2) {
            return true;
        }
        int[] tmp = Arrays.copyOfRange(arr,left,right+1);
        Arrays.sort(tmp);
        int diff = tmp[1] - tmp[0];
        for(int i = 1; i < tmp.length; i++) {
            if(tmp[i] - tmp[i-1] != diff) {
                return false;
            }
        }
        return true;
    }

    /**
     * 1631. 最小体力消耗路径
     * @param heights
     * @return
     */
}
