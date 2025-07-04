import cv2
import numpy as np


cap = cv2.VideoCapture(0)  # Open the default camera


while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break
    else:
        frame = cv2.flip(frame, 1) #horizontal (like a phone is)
    

    #cv2.circle(frame, (638,357), 3, (0, 255, 0), 5)
    eyes = [[130, 553, 332], [243, 617, 341], [27, 577, 318], [23, 588, 349], [463, 675, 338], [359, 738, 324], [257, 713, 313], [253, 705, 344]]
    fores = [[564.0, 172.0], [564.0, 178.0], [560.0, 182.0], [556.0, 182.0], [554.0, 184.0], [552.0, 184.0], [550.0, 186.0], [548.0, 186.0], [546.0, 188.0], [544.0, 188.0], [540.0, 192.0], [538.0, 192.0], [532.0, 198.0], [524.0, 198.0], [524.0, 290.0], [552.0, 290.0], [554.0, 288.0], [558.0, 288.0], [560.0, 290.0], [578.0, 290.0], [580.0, 292.0], [594.0, 292.0], [596.0, 294.0], [604.0, 294.0], [606.0, 296.0], [612.0, 296.0], [614.0, 298.0], [618.0, 298.0], [620.0, 300.0], [620.0, 306.0], [656.0, 306.0], [656.0, 300.0], [660.0, 296.0], [666.0, 296.0], [668.0, 294.0], [672.0, 294.0], [674.0, 292.0], [680.0, 292.0], [682.0, 290.0], [686.0, 290.0], [688.0, 288.0], [692.0, 288.0], [694.0, 286.0], [698.0, 286.0], [700.0, 284.0], [704.0, 284.0], [706.0, 282.0], [712.0, 282.0], [714.0, 280.0], [726.0, 280.0], [728.0, 278.0], [740.0, 278.0], [744.0, 274.0], [744.0, 268.0], [746.0, 266.0], [746.0, 258.0], [748.0, 256.0], [748.0, 246.0], [750.0, 244.0], [750.0, 228.0], [752.0, 226.0], [752.0, 204.0], [750.0, 202.0], [750.0, 196.0], [748.0, 194.0], [748.0, 190.0], [746.0, 188.0], [746.0, 186.0], [738.0, 178.0], [738.0, 172.0], [642.0, 172.0], [642.0, 178.0], [640.0, 180.0], [638.0, 180.0], [636.0, 182.0], [628.0, 182.0], [624.0, 178.0], [624.0, 172.0]]

    for point in fores:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)
    for point in eyes:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)
    cv2.circle(frame, (628, 182), 3, (255, 0, 0), 5)


    cv2.imshow('DailyTefillin', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()