// 逻辑运算符
// 逻辑迷宫

func doIt() {
    moveForward()

    //第一个岔路
    if(isOnClosedSwitch && isOnGem){
        turnRight()
    }
    //草地尽头
    if isBlocked && isOnGem {
        turnLeft()
        turnLeft()
    }
    //从草地回来后
    if isOnOpenSwitch && isBlocked {
        turnRight()
    }
    //有开关且处于关闭，没宝石
    if isOnClosedSwitch && !isOnGem {
        turnLeft()
    }
    
    
    if(isOnClosedSwitch){
        toggleSwitch()
    }
    if(isOnGem){
        collectGem()
    }
    
}

for i in 1...18{
    doIt()
}

