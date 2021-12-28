package LeetCodeExcise;

import java.util.Scanner;

public class DuiLie {
    public static void main(String[] args) {
        ArrayQueue arrayQueue = new ArrayQueue(3);
        char key = ' ';
        Scanner scanner = new Scanner(System.in);
        boolean loop = true;
        while (loop){
            System.out.println("s(show):显示队列");
            System.out.println("e(exit):退出队列");
            System.out.println("a(add):添加数据到队列");
            System.out.println("g(get):从队列取出数据");
            System.out.println("h(head):查看队列头部数据");
            key = scanner.next().charAt(0);
            switch (key){
                case 's':
                    arrayQueue.showData();
                    break;
                case 'a':
                    System.out.println("请输入一个整数");
                    int value = scanner.nextInt();
                    arrayQueue.addData(value);
                    break;
                case 'g':
                    try{
                        System.out.printf("取出的数据为%d\n",arrayQueue.getData());
                    }catch (Exception e){
                        System.out.println(e.getMessage());
                    }
                    break;
                case 'h':
                    try{
                        System.out.printf("取出的数据为%d\n",arrayQueue.headData());
                    }catch (Exception e){
                        System.out.println(e.getMessage());
                    }
                    break;
                case 'e':
                    scanner.close();
                    loop=false;
                    break;
                default:
                    break;
            }
        }
        System.out.println("程序退出");

    }

}
class ArrayQueue{
    private int max_size;
    private int front;
    private int rear;
    private int[] arr;

    public ArrayQueue(int max_size) {
        this.max_size = max_size;
        arr =  new int[max_size];
        front = -1;
        rear = -1;
    }
    public boolean isEmpty(){
        return front==rear;

    }
    public boolean isFull(){
       return rear==max_size-1;
    }
    public void addData(int n){
/*        if(isFull()){
            System.out.println("队列满，添加数据失败");
            return;
        }*/
        rear++;
        if(rear > max_size-1){
            rear=rear%(max_size-1)-1;
        }
        arr[rear] = n;
    }
    public int getData(){
        if(isEmpty()){
            throw new RuntimeException("队列为空，获取数据失败");
        }
        front++;
        if(front > max_size-1){
            front=front%(max_size-1)-1;
        }
        return arr[front];
    }
    public void showData(){
        if(isEmpty()){
            System.out.println("队列为空，无法显示数据");
        }
        for(int i=0 ;i<arr.length;i++){
            System.out.printf("arr[%d]=%d\n",i,arr[i]);
        }
    }
    public int headData(){
        if(isEmpty()){
            throw new RuntimeException("队列为空，无法显示队列首数据");
        }
        return arr[front+1];
    }
    public int rearData(){
        if(isEmpty()){
            throw new RuntimeException("队列为空，无法显示队列尾部数据");

        }
        return arr[rear];
    }
}
