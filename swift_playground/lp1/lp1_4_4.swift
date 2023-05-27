// 条件代码
// 满足条件时向上爬

for i in 1 ... 13 {
    if isOnGem {
        collectGem()
        turnLeft()
        moveForward()
    } else {
        moveForward()
    }
}
