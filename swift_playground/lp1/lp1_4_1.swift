// 条件代码
// 检查开关

func checkToggleSwitch() {
    moveForward()
    if isOnClosedSwitch {
        toggleSwitch()
    }
}

moveForward()
for i in 1 ... 3 {
    checkToggleSwitch()
}






