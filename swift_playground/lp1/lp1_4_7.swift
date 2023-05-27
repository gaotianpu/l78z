// 条件代码
// 决策树
func turnAround(){
    turnLeft()
    turnLeft()
}

func solveRightSide(){
    turnRight()
    for i in 1 ... 6{
        moveForward()
        if isOnGem{
            collectGem()
            turnAround()
            for i in 1 ... 4 {
                moveForward()
                if isBlocked{
                    turnRight()
                }
            }
            break
        } else if isBlocked{
            turnLeft()
        }
    } 
}

func solveLeftSide(){
    turnLeft()
    moveForward()
    if isOnGem{
        collectGem()
        turnAround()
        moveForward()
    }
    turnLeft()
}

for i in 1 ... 6{
    moveForward()
    if isOnGem{
        collectGem()
        solveRightSide()
        
    }else if isOnClosedSwitch{
        toggleSwitch()
        solveLeftSide()
    } 
}

