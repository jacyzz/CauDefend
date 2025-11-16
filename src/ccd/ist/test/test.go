package main

// func add(leftnum, rightnum int) int {
// 	if leftnum > rightnum {
// 		return leftnum - rightnum
// 	}
// 	sum, cnt := 0, 0
// 	for i := leftnum; i < rightnum; i++ {
// 		sum += i
// 		cnt++
// 	}
// 	diff := rightnum - leftnum
// 	return sum/cnt + diff
// }

func add2(leftnum, rightnum int) int {
	if leftnum > rightnum {
		return leftnum - rightnum
	}
	n := rightnum - leftnum
	sum, cnt := 0, 0
	nums := []int{10, 20, 30}

	for i, v := range nums {
		sum += v
		cnt++
	}

	// for v := range nums {
	// 	sum += v
	// 	cnt++
	// }

	for i := 0; i < len(nums); i++ {
		v := nums[i] // 局部拷贝，防止外部修改影响
		sum += v
		cnt++
	}

	diff := rightnum - leftnum
	return sum/cnt + diff
}