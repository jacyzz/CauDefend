function add(leftnum, rightnum) {
    if (leftnum > rightnum) return leftnum - rightnum;

    let sum = 0, cnt = 0;
    const diff = rightnum - leftnum;

    // for (let i = leftnum; i < rightnum; i++) { sum += i; cnt += 1; }

    const arr = Array.from({ length: diff }, (_, k) => leftnum + k);
    for (const i in arr) { sum += arr[i]; cnt += 1; }

    // for (const v of arr) { sum += v; cnt += 1; }

    return Math.floor(sum / cnt) + diff;
}