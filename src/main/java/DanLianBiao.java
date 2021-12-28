public class DanLianBiao {
    public static void main(String[] args) {
        HeroNode hero1 = new HeroNode(1,"宋江","及时雨");
        HeroNode hero2 = new HeroNode(2,"卢俊义","玉麒麟");
        HeroNode hero3 = new HeroNode(3,"吴用","智多星");
        HeroNode hero4 = new HeroNode(4,"林冲","豹子头");
        SingleLinedList singleLinedList = new SingleLinedList();
        singleLinedList.add(hero1);
        singleLinedList.add(hero2);
        singleLinedList.add(hero3);
        singleLinedList.add(hero4);
        singleLinedList.list();

    }
}
//定义一个单链表
class SingleLinedList{
    HeroNode head = new HeroNode(0,"","");

    public void add(HeroNode heroNode){
        HeroNode temp = head;
        while (true){
            if(temp.next==null){
                break;
            }else{
                temp = temp.next;
            }
        }
        temp.next = heroNode;
    }

    public void list(){
        if(head.next==null){
            System.out.println("链表为空！");
            return;
        }
        HeroNode temp = head.next;
        while (true){
            if(temp==null){
                break;
            }
            System.out.println(temp.toString());
            temp = temp.next;
        }
    }
}
class HeroNode{
    public int no;
    public String name;
    public String nickName;
    public HeroNode next;

    public HeroNode(int no, String name, String nickName) {
        this.no = no;
        this.name = name;
        this.nickName = nickName;
    }

    @Override
    public String toString() {
        return "HeroNode{" +
                "no=" + no +
                ", name='" + name + '\'' +
                ", nickName='" + nickName + '\'' +
                '}';
    }
}
