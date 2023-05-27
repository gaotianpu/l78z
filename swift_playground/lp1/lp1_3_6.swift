// For循环
// 宝石农场

func moveCollect(){
    for i in 1 ... 2 {
        moveForward()
        collectGem()
    }
}

func moveSwitch(){
    for i in 1 ... 2 {
    moveForward()
    toggleSwitch()
    }
}

func goBack(){
    turnLeft()
    turnLeft()
    moveForward()
    moveForward() 
}

func oneRow(){
    turnLeft()
    
    moveSwitch()
    goBack() 
    moveCollect()
    goBack()
    
    turnRight()
    moveForward()
    
}


for i in 1...3{
    oneRow()
}



