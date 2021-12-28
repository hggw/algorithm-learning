package LeetCodeExcise;

public class ParkingSystem {
    int[] carMount;

    public ParkingSystem(int big, int medium, int small) {
        carMount = new int[]{big,medium,small};
    }

    public boolean addCar(int carType) {
        switch (carType) {
            case 1 :
                if(carMount[0] <= 0) {
                    return false;
                }else {
                    carMount[0]--;
                    break;
                }
            case 2 :
                if(carMount[1] <= 0) {
                    return false;
                }else {
                    carMount[1]--;
                    break;
                }
            case 3 :
                if(carMount[2] <= 0) {
                    return false;
                }else {
                    carMount[2]--;
                    break;
                }
        }
        return true;
    }
}
