// 条件代码
// 使用 else if



func checkCollectToggle() {
    moveForward()
    if isOnGem {
        collectGem()
    }else if isOnClosedSwitch{
        toggleSwitch() 
    }
}

for i in 1 ... 2 {
    checkCollectToggle()
}


