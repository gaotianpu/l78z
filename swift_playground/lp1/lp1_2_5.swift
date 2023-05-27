// 函数
// 嵌套模式

func turnAround(){
    turnLeft()
    turnLeft()
}

func solveStair(){
    moveForward()
    collectGem()
    turnAround()
    moveForward()
}

solveStair()
solveStair()
turnLeft()
solveStair()
solveStair()
