def add(leftnum, rightnum):
    if leftnum > rightnum:
        return leftnum - rightnum

    sum = 0
    cnt = 0
    for i in range(leftnum, rightnum):
        sum += i
        cnt += 1

    diff = rightnum - leftnum
    return sum / cnt + diff