public class XiSuShuZu {
    public static void main(String[] args) {
        //0,1,2分别表示无子，白子，黑子
        int[][] chessArr = new int[11][11];
        chessArr[1][2]=1;
        chessArr[2][4]=2;
        //将二位数组转为稀疏数组
        int count = 0;
        for(int i=0;i<11;i++){
            for(int j=0;j<11;j++){
                if(chessArr[i][j] != 0){
                    count += 1;
                }
            }
        }
        int[][] sparseArr = new int[count+1][3];
        sparseArr[0][0]=11;
        sparseArr[0][1]=11;
        sparseArr[0][2]=count;
        int temp = 1;
        for(int i=0;i<11;i++){
            for(int j=0;j<11;j++){
                if(chessArr[i][j]!=0){
                    sparseArr[temp][0] = i;
                    sparseArr[temp][1] = j;
                    sparseArr[temp][2] = chessArr[i][j];
                    temp += 1;
                }
            }
        }
        for(int i=0;i<=count;i++){
            for(int j=0;j<3;j++){
                System.out.print(sparseArr[i][j]);
            }
            System.out.println();
        }

        //稀疏数组转二维数组
        int[][] orgArr = new int[sparseArr[0][0]][sparseArr[0][1]];
        for(int i = 1;i<sparseArr.length;i++){
            orgArr[sparseArr[i][0]][sparseArr[i][1]] = sparseArr[i][2];
        }
        for(int i =0;i<orgArr.length;i++){
            for(int j =0;j<orgArr[0].length;j++){
                System.out.print(orgArr[i][j]);
            }
            System.out.println();
        }

    }
}
