/*
For循环 席卷四处
*/


func turnAround(){
    turnLeft() 
    turnLeft()
    moveForward()  
}

func moveCollect(){
    moveForward()
    collectGem() 
}

func one(){
    moveCollect()
    turnLeft()
    
    moveCollect()
    turnAround() 
    
    turnLeft()
    
    moveCollect() 
    turnAround() 
    
    turnLeft()
    moveCollect()
    moveForward()    
}

for i in 1 ... 4 {
    one()
}

