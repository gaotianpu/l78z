// 逻辑运算符
// 检查这个或那个

for i in 1 ... 12 { 
    moveForward()
    if isBlocked || isBlockedLeft {
        turnRight()
    }
    if isOnGem{
        collectGem()
    }
}