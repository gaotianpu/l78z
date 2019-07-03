import numpy as np  
import cv2    
              
def draw_circle(event, x, y, flags, param):    
    if event == cv2.EVENT_LBUTTONDBLCLK:        
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)    
  
# 新建图像窗口并将窗口与回调函数绑定
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')      
cv2.setMouseCallback('image', draw_circle)    
        
while (1):     
# 显示图像并且按键盘上的**ESC**键即可关闭窗口  
    cv2.imshow('image', img)    
    if cv2.waitKey(20) & 0xFF == 27:       
        break    
# 最后销毁窗口  
cv2.destroyAllWindows()