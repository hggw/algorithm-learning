import java.util.Scanner;

public class HuanXingDuiLie {
    public static void main(String[] args) {
        System.out.println("测试环形队列的案例~~~~");
        CircleArrayQueue arrayQueue = new CircleArrayQueue(4);
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
class CircleArrayQueue{
    private int max_size;
    private int front;
    private int rear;
    private int[] arr;

    public CircleArrayQueue(int maxsize) {
        this.max_size = maxsize;
        arr =  new int[max_size];
    }
    public boolean isEmpty(){
        return front==rear;

    }
    public boolean isFull(){
        return (rear+1)%max_size==front;
    }
    public void addData(int n){
        if(isFull()){
            System.out.println("队列满，添加数据失败");
            return;
        }
        arr[rear] = n;
        rear=(rear+1)%max_size;
    }
    public int getData(){
        if(isEmpty()){
            throw new RuntimeException("队列为空，获取数据失败");
        }
        int tmp=arr[front];
        front=(front+1)%max_size;
        return tmp;
    }
    public void showData(){
        if(isEmpty()){
            System.out.println("队列为空，无法显示数据");
        }
        for(int i=front ;i<front+size();i++){
            System.out.printf("arr[%d]=%d\n",i%max_size,arr[i%max_size]);
        }
    }
    public int size(){
        //这里必须要把max_size加到括号内，保证括号内的值不会是负数
        return (rear+max_size-front)%max_size;
    }
    public int headData(){
        if(isEmpty()){
            throw new RuntimeException("队列为空，无法显示队列首数据");
        }
        return arr[front];
    }
}
