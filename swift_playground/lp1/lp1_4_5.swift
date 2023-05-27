// 条件代码
// 定义更巧妙的函数

func collectOrToggle(){
    moveForward()
    if isOnGem{
        collectGem()
    }else if isOnClosedSwitch{
        toggleSwitch()
    } 
}

func collectOneRow(){
    for i in 1...4{
        collectOrToggle()
    }
}


collectOneRow()

turnLeft()
moveForward()
moveForward()
turnLeft()

collectOneRow()

turnRight()
moveForward()
turnRight()

collectOneRow()