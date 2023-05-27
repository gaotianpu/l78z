// While循环
// 转身

for i in 1 ... 11 {
    moveForward()
    if !isOnGem {
        turnRight()
    }
    if isOnGem {
        collectGem()
    }
    if isBlocked{
        turnLeft()
    }
    
}
