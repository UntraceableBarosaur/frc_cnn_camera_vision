import cv2
import numpy as np
from PIL import Image

# Pickup the TRAINING DATA WHERE WE LEFT OFF
file_name = 'test.npy'
loaded_table = np.load(file_name, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
print(loaded_table[1])
print(len(loaded_table[0]))
print(loaded_table)
output_table = loaded_table


# START VIDEO CAPTURE
cap = cv2.VideoCapture(1)

# SET SOME BASIC START VALUES
font = cv2.FONT_HERSHEY_SIMPLEX
hue = 0
saturation = 0
value = 0
u_hue = 255
u_saturation = 255
u_value = 255
training_active = False


# SET THE BASE COLOR VALUES 
lower_green = np.array([0,0,0])
upper_green = np.array([255,255,255])
#lower_green = np.array([45,78,40])
#upper_green = np.array([120,255,255])


# IMAGE CONFIGURATION DATA
edge_1 = 100
edge_2 = 200
unit_answer = 3
crop_ratio = 0.25
blur_severity = 15
img_height = 14
img_width = 56
img_pixels = 784


while(1):
    _, frame = cap.read()

    lower_green = np.array([hue,saturation,value])
    upper_green = np.array([u_hue,u_saturation,u_value])

    
    frame = frame[int(frame.shape[0]*crop_ratio):int(frame.shape[0]*(1-crop_ratio)),0:int(frame.shape[1])]
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    edges = cv2.Canny(frame,edge_1,edge_2)
    #ret, mask2 = cv2.threshold(edges, 220, 255, cv2.THRESH_BINARY_INV)
    #mask2_inv = cv2.bitwise_not(mask2)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame,frame, mask=mask)

    bgr_res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray_res = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2GRAY)
    output = cv2.add(gray_res,edges)
    res = cv2.medianBlur(output,blur_severity)
    
    resized = cv2.resize(res, (int(img_width),int(img_height)), interpolation = cv2.INTER_AREA)
    gray = resized
    if(training_active == True):
        cv2.putText(res,'TRAINING IN PROGRESS (SPACE TO CANCEL)',(10,200), font, 0.75, (200,255,155), 2, cv2.LINE_AA)

    cv2.putText(res,('lower_hue (Q+A): '+str(hue)),(0,15), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    cv2.putText(res,('lower_saturation (W+S): '+str(saturation)),(0,45), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    cv2.putText(res,('lower_value (E+D): '+str(value)),(0,75), font, 0.5, (200,255,155), 1, cv2.LINE_AA)

    cv2.putText(res,('upper_hue (R+F): '+str(u_hue)),(0,30), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    cv2.putText(res,('upper_saturation (T+G): '+str(u_saturation)),(0,60), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    cv2.putText(res,('upper_value (Y+H): '+str(u_value)),(0,90), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    
    #Figure Out Which Side We ARE ON
    section_one = int(cv2.countNonZero(gray[0:int(gray.shape[0]),0:int(gray.shape[1]*0.45)]))
    section_zero = int(cv2.countNonZero(gray[0:int(gray.shape[0]),int(gray.shape[1]*0.45):int(gray.shape[1]*0.55)]))
    section_two = int(cv2.countNonZero(gray[0:int(gray.shape[0]),int(gray.shape[1]*0.55):int(gray.shape[1])]))

    print(section_zero)
    print(section_one)
    print(section_two)

    if(section_one>section_zero and section_one>section_two):
        cv2.putText(res,'LEFT',(0,150), font, 0.5, (200,255,155), 2, cv2.LINE_AA)
        unit_answer = 1        
    elif(section_two>section_zero and section_two>section_one):
        cv2.putText(res,'RIGHT',(0,150), font, 0.5, (200,255,155), 2, cv2.LINE_AA)
        unit_answer = 2
    elif(section_zero>section_one and section_zero>section_two):
        cv2.putText(res,'MIDDLE',(0,150), font, 0.5, (200,255,155), 2, cv2.LINE_AA)
        unit_answer = 0
    else:
        unit_answer = 3

    cv2.imshow('Averaging',res)
    cv2.imshow('Resized',resized)
    #cv2.imshow('res',output)
    
    if(training_active == True and not(unit_answer==3)):    
        #FORMAT THE OUTPUT TABLE
        temp_data = [np.float32(np.array(resized).ravel()/255),unit_answer]
        output_table = np.vstack([output_table,temp_data])
        print('Answer: '+str(unit_answer))
        print('Data Length: '+str(len(temp_data[0])))
        print('Output Shape: '+str(len(temp_data)))
        print('File Shape: '+str(output_table.shape))


    #CHECK FOR KEY INTERRUPTS
    k = cv2.waitKey(5) & 0xFF

    if(k == ord('q') and hue < 255):
        hue += 1
    elif(k == ord('a') and hue > 0):
        hue -= 1
    elif(k == ord('w') and saturation < 255):
        saturation += 1
    elif(k == ord('s') and saturation > 0):
        saturation -= 1
    elif(k == ord('e') and value < 255):
        value += 1
    elif(k == ord('d') and value > 0):
        value -= 1
    elif(k == ord('r') and u_hue < 255):
        u_hue += 1
    elif(k == ord('f') and u_hue > 0):
        u_hue -= 1
    elif(k == ord('t') and u_saturation < 255):
        u_saturation += 1
    elif(k == ord('g') and u_saturation > 0):
        u_saturation -= 1
    elif(k == ord('y') and u_value < 255):
        u_value += 1
    elif(k == ord('h') and u_value > 0):
        u_value -= 1
    elif(k == ord(' ')):
        if(training_active == True):
            training_active = False
        else:
            training_active = True

    
    if k == 27:
        #SAVE THE OUTPUT TABLE
        #np.save(file_name,output_table, allow_pickle=True, fix_imports=True)
        break


cv2.destroyAllWindows()
cap.release()

