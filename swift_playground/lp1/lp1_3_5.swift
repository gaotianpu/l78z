// For循环
// 拓展技能

func tunAround(){
    turnLeft()
    turnLeft()
}

func forword7(){
    for i in 1 ... 7 {
        moveForward()
    }
}

func toggleOne(){
    moveForward()
    moveForward()
    turnRight()
    forword7()
        
    toggleSwitch()
    tunAround()
    forword7()

    turnRight()   
}


for i in 1 ... 3 {
    toggleOne()
}