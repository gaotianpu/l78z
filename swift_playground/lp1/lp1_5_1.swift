// 逻辑运算符
// 使用"非"运算符

for i in 1 ... 4 {
    moveForward()
    if !isOnGem{
        turnLeft()
        moveForward()
        moveForward()
        collectGem()
        turnLeft()
        turnLeft()
        moveForward()
        moveForward()
        turnLeft()
    } else {
        collectGem()
    } 
}