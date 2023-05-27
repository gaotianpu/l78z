// 逻辑运算符
// 检查这个与那个

for i in 1 ... 13 {
    moveForward()
    
    if isOnGem && isBlockedLeft {
        collectGem()
        turnRight()
    } else if isOnGem{
        collectGem()
    }
    
    if isBlocked && isOnClosedSwitch{
        toggleSwitch()
        turnLeft()
        turnLeft()
        moveForward()
        moveForward()
        turnRight()
    }
    
}