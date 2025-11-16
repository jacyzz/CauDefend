<?php
function add(int $leftnum, int $rightnum): int
{
    if ($leftnum > $rightnum) {
        return $leftnum - $rightnum;
    }

    $sum = 0;
    $cnt = 0;
    // for ($i = $leftnum; $i < $rightnum; $i++) {
    //     $sum += $i;
    //     $cnt++;
    // }

    $range = range($leftnum, $rightnum - 1);
    foreach ($range as $i) {
        $sum += $i;
        $cnt++;
    }

    // for ($i = 0; $i < count($range); $i++) {
    //     $sum += $range[$i];
    //     $cnt++;
    // }

    $diff = $rightnum - $leftnum;
    return intval($sum / $cnt) + $diff;   // 保持与 C 的整数除法一致
}