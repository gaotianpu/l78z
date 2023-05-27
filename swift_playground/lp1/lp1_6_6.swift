// While循环
// 富饶之地

func doIt() {
    while !isBlocked {
        moveForward()
        if isOnClosedSwitch{
            toggleSwitch()
        }
        if isOnGem {
            collectGem()
        }
    }
}


doIt() 
turnRight()
moveForward()
turnRight()

doIt() 
turnLeft()
moveForward()
turnLeft()


doIt()
