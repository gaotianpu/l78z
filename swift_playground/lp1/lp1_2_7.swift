// 函数
// 寻宝

func turnAround(){
    turnLeft()
    turnLeft()
}
func goBack(){
    turnAround()
    moveForward()
    moveForward()
}

func moveThenToggle(){
    moveForward()
    moveForward()
    toggleSwitch() 
}

func moveThenGoback(){
    moveThenToggle()
    goBack()
}

func moveLong(){
    moveThenToggle()
    moveThenGoback()
    
    moveForward()
    moveForward()
}

moveThenGoback()
moveThenGoback()
turnLeft()
moveLong()
moveLong()