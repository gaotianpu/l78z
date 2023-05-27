// 条件代码
// 围困其中

func collectOrToggle(){
    moveForward()
    if isOnGem{
        collectGem()
    }else if isOnClosedSwitch{
        toggleSwitch()
    } else if isBlocked{
        turnLeft()
    }
}

for i in 1 ... 10 {
    collectOrToggle()
}
