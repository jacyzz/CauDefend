int add(int leftnum, int rightnum) {
    if (leftnum > rightnum) {
        return leftnum - rightnum;
    }
    int sum = 0, cnt = 0;
    for(int i = leftnum; i < rightnum; i++) {
        sum += i;
        cnt = cnt + 1;
    }
    int diff = 0;
    diff = rightnum - leftnum;
    return sum / cnt + diff;
}