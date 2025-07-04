
import cv2
import numpy as np
from ultralytics import YOLO
import detection_code as LM
from collections import deque

face_detector = LM.Face()  # detect face with variable
LEFT_EYE_IDX  = [33, 133]    # outer & inner corners of the left eye
RIGHT_EYE_IDX = [362, 263]   # outer & inner corners of the right eye

def detect_face(image): #isn't used; but it would detect facial features to get referance points for tefillin
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_detector.Find_Points(image, imgRGB)

# at top, define which mesh indices to grab

def detect_eyes(image):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pts = face_detector.Find_Points(image, imgRGB)
    if not pts:
        return [None, None]

    # grab the two corner points for each eye
    left  = [pts[i] for i in LEFT_EYE_IDX]
    right = [pts[i] for i in RIGHT_EYE_IDX]
    return [left, right]


def detect_hairline(image):
    hairline_model = YOLO('/Users/koviressler/Desktop/DailyTefillin/hairline_detection/hairlineAI.pt')
    hairlines = hairline_model(image)
    #print("hairlines:", hairlines) #only use to debug
    if len(hairlines) > 0 and hairlines[0].masks is not None: #if there is a hairline detected
        # Get the mask with the highest confidence
            # Turn YOLO object into the coordinates
        all_points = []
        for result in hairlines:
            if result.masks is not None:
                for mask in result.masks.xy:
                    all_points.extend(mask.tolist()) #add hairline cords to list.
        return all_points
    else:
        return None #no hairlines detected


def detect_tefillin(image):
    # Run inference on the image
    model = YOLO('/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/runs/detect/train6/weights/best.pt') 
    results = model(image)
    
    return results

def paint_tefillin(image, results):
    # Process and draw results
    for result in results: #for every point in list given
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, confidence in zip(boxes, confidences):
            if confidence > 0.7:  # Adjust this threshold as needed
                x1, y1, x2, y2 = map(int, box[:4]) #get every coordinate for each points and draw
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'Tefillin: {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def lowest_hairline_point(eye_cords, forehead_cords): #the bottom of their tefillin must be above this point
    if not eye_cords[0] or not eye_cords[1] or not forehead_cords:
        return None
    
    # Extract x-coordinates of the eyes
    left_x = eye_cords[0][1][1]  # right-most x value of left eye
    right_x = eye_cords[1][0][1]  # left-most x value of right eye
    #middle of tefillin must be in between these two points
    
    
    imp_points = []
    for i in range(len(forehead_cords)): #only take points between eyes. we don't care about side hair, and this accounts for widows peak
        if left_x < forehead_cords[i][0] < right_x:
            imp_points.append(forehead_cords[i])
    # print("forehead:", all_points)
    # print("eyes:", eye_cords)

    if not imp_points: #if not detected (make them retake)
        return None
    

    #artificially add in ideal point (if their forehead is completely flat)
    forehead_cords.sort(key=lambda p: p[1]) #sort from highest y-value to lowest (x is disregarded)
    left_most_point = None
    right_most_point = None
    for point in forehead_cords: #from highest y-value to lowest
        if point[0] < left_x and not left_most_point: #highest point on left side of face (doesn't include points between eyes)
            left_most_point = point
        elif point[0] > right_x and not right_most_point:
            right_most_point = point

        if left_most_point and right_most_point:
            break
    
    #find midpoint y value
    if left_most_point and right_most_point:
        artif_point = [((left_x + right_x) / 2), ((left_most_point[1] + right_most_point[1]) / 2)]
        imp_points.append(artif_point)


    # Sort points by y-coordinate
    imp_points.sort(key=lambda p: p[1])
    #print(imp_points)

    #find the difference between each y-value 
    big_dif = [0, None] #[difference, slot]
    for i in range(len(imp_points) - 1): #for every point in hairline list
        if i != 0 and imp_points[i][1] - imp_points[i-1][1] > big_dif[0]: #want i to start at second number
            big_dif[0] = imp_points[i][1] - imp_points[i-1][1]
            big_dif[1] = i #if the jump in y values is the biggest yet, keep track (it must be the start of the hairline)
    
    

    hairline_cords = imp_points[:big_dif[1]] #take all points on hairline
    lowest_point = hairline_cords[len(hairline_cords)-1] #last cord (biggest y) is furthest down
    return lowest_point

def paint_hairline(hairline_points):
    if not hairline_points:
        return
    for point in hairline_points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

def paint_eyes(eye_points):
    if eye_points[0] and eye_points[1]:
        for i in range(2):
            for point in eye_points[i]:
                cv2.circle(frame, (int(point[1]), int(point[2])), 3, (0, 255, 0), -1)

def detect_face_and_paint(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_detector.Drawing(img, imgRGB, color=(0, 110, 100), thickness=3, circle_radius=3)
    face_detector.Find_Points(img, imgRGB)

def get_lowest_hairline_point(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None

    #for displaying
    display_image = image.copy()
    eye_coordinates = detect_eyes(image)
    hairline_points = detect_hairline(image)

    # Find the lowest hairline point
    lowest_point = lowest_hairline_point(eye_coordinates, hairline_points)

    if lowest_point:
        # Draw the lowest point on the display image
        #cv2.circle(display_image, (int(lowest_point[0]), int(lowest_point[1])), 5, (0, 0, 255), -1)
        height, width = display_image.shape[:2]
        y = int(lowest_point[1])
        cv2.line(display_image, (0, y), (width, y), (255, 0, 0), 2)  # Blue line, 2 pixels thick

        # Display the image
        cv2.imshow("Image with Lowest Hairline Point", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return lowest_point

def face_features(image):
    face_points = detect_face(image)
    if face_points:
        return [face_points[152], face_points[2]] #chin, nose
    return [None, None]

def paint_face_features(face_points):
    if face_points[0] and face_points[1]:
        for point in face_points:
            cv2.circle(frame, (point[1], point[2]), 3, (0, 255, 0), -1)

def detect_head_pose(image, face_landmarks):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (face_landmarks[30][1:3]),     # Nose tip
        (face_landmarks[152][1:3]),    # Chin
        (face_landmarks[36][1:3]),     # Left eye left corner
        (face_landmarks[45][1:3]),     # Right eye right corner
        (face_landmarks[48][1:3]),     # Left Mouth corner
        (face_landmarks[54][1:3])      # Right mouth corner
    ], dtype="double")

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angles
    #returns a list of 3 lists
    # Pitch: left right
    # Yaw: up down
    # Roll: side side

def paint_head_pose(image, euler_angles):
    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def get_reference_pose(image_path): #helper function to find head position using path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None
    
    face_points = detect_face(image)
    if face_points:
        x = detect_head_pose(image, face_points)
        print("Reference Pose:", x)
        return x
    return None

def compare_poses(current_pose, reference_pose, threshold=4.0): #returns the biggest difference in head position (how to fix it)
    pitch_diff = current_pose[0][0] - reference_pose[0][0] #left + right
    yaw_diff = current_pose[1][0] - reference_pose[1][0]
    roll_diff = current_pose[2][0] - reference_pose[2][0]
    
    guidance = []
    if abs(pitch_diff) > threshold:
        guidance.append(("Turn " + ("right" if pitch_diff > 0 else "left"), abs(pitch_diff)))
    if abs(roll_diff) > threshold:
        guidance.append(("Tilt head " + ("left" if roll_diff > 0 else "right"), abs(roll_diff)))
    
    # Sort guidance by magnitude of difference
    guidance.sort(key=lambda x: x[1], reverse=True)
    
    if abs(yaw_diff) > threshold:
        guidance.insert(0, ("Turn " + ("down" if yaw_diff > 0 else "up"), abs(yaw_diff)))

    return guidance

def smooth_poses(pose_queue):
    if len(pose_queue) == 0:
        return None
    poses = np.array(pose_queue)
    return np.mean(poses, axis=0)

def paint_pose_guidance(image, guidance):
    if not guidance:
        cv2.putText(image, "Position matched!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:

        message = guidance[0][0]  # Only show the most significant adjustment
        cv2.putText(image, message, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def distance(point1, point2): #gets dis between 2 points - list of [x,y,...]
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def ratio(bottom, top): #returns the number the chin pos should be multiplied by to get hairline
    return (top/bottom) #just multiply bottom by this to get the new hairline

def draw_hairline(frame):
    face_points = detect_face(frame)
    if face_points and ref_ratio is not None:
        current_pose = detect_head_pose(frame, face_points)
        pose_queue.append(current_pose)
        
        smoothed_pose = smooth_poses(pose_queue)
        if smoothed_pose is not None:
            paint_head_pose(frame, smoothed_pose)
            guidance = compare_poses(smoothed_pose, reference_pose)
            paint_pose_guidance(frame, guidance)

            if not guidance:  # if the head is matched correctly
                frame_features = face_features(frame)
                if frame_features[0] and frame_features[1]:
                    chin = frame_features[0][1:3]
                    nose = frame_features[1][1:3]
                    frame_bottom_dis = distance(chin, nose)
                    
                    # Calculate predicted hairline position
                    predicted_top_dis = frame_bottom_dis * ref_ratio
                    hairline_vector = np.array(nose) - np.array(chin)
                    hairline_vector = hairline_vector / np.linalg.norm(hairline_vector)
                    predicted_hairline = nose + (hairline_vector * predicted_top_dis)
                    
                    # Draw hairline
                    cv2.line(frame, (0, int(predicted_hairline[1])), (frame.shape[1], int(predicted_hairline[1])), (255, 0, 0), 2)
                    
                    # Draw feature points
                    cv2.circle(frame, (int(chin[0]), int(chin[1])), 5, (0, 255, 0), -1)  # Chin
                    cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), -1)  # Nose
                    cv2.circle(frame, (int(predicted_hairline[0]), int(predicted_hairline[1])), 5, (255, 255, 0), -1)  # Predicted hairline

def evaluate_tefillin_placement(image, reference_image):
    # Step 1: Compute reference ratio (distance from chin to nose, nose to hairline point)
    ref_img = cv2.imread(reference_image)
    if ref_img is None:
        return False, f"Reference image not found at {reference_image}"
    ref_eyes = detect_eyes(ref_img)
    ref_hairline = detect_hairline(ref_img)
    ref_features = face_features(ref_img)


    if not ref_hairline or not ref_eyes or not ref_features[0] or not ref_features[1]:
        return False, "Reference image missing chin, nose, eyes, or hairline."

    ref_hairpoint = lowest_hairline_point(ref_eyes, ref_hairline)
    if not ref_hairpoint:
        return False, "Reference hairline point not found."

    ref_bottom_dis = distance(ref_features[0][1:3], ref_features[1][1:3])  # chin to nose
    ref_top_dis = distance(ref_features[1][1:3], ref_hairpoint)  # nose to hairline
    ref_ratio = ref_top_dis / ref_bottom_dis

    # Step 2: Analyze test image
    eye_coords = detect_eyes(image)
    hairline_points = detect_hairline(image)
    tefillin_results = detect_tefillin(image)
    face_points = detect_face(image)

    if not eye_coords or not eye_coords[0] or not eye_coords[1]:
        print("EYE CORDS:")
        print(eye_coords)
        exit()
        #return False, "Eyes not detected in test image."
    if not hairline_points:
        return False, "Hairline not detected in test image."
    if not tefillin_results or len(tefillin_results) == 0 or tefillin_results[0].boxes is None:
        return False, "Tefillin not detected in test image."
    if not face_points:
        return False, "Face points not detected in test image."

    # Step 3: Predict target hairline line (based on chin/nose and ref ratio)
    chin = face_points[152][1:3]
    nose = face_points[2][1:3]
    bottom_dis = distance(chin, nose)
    top_dis = bottom_dis * ref_ratio
    vector = np.array(nose) - np.array(chin)
    vector = vector / np.linalg.norm(vector)
    predicted_hairline = np.array(nose) + vector * top_dis
    hairline_y = predicted_hairline[1]

    # Step 4: Get horizontal eye bounds
    left_x = eye_coords[0][1][1]
    right_x = eye_coords[1][0][1]

    # Step 5: Evaluate tefillin
    for result in tefillin_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            if confidence < 0.7:
                continue
            x1, y1, x2, y2 = map(int, box)
            mid_x = (x1 + x2) // 2
            bottom_y = y2

            # Horizontal centering check
            if not (left_x < mid_x < right_x):
                if mid_x <= left_x:
                    return False, "Move tefillin right to center it between your eyes."
                elif mid_x >= right_x:
                    return False, "Move tefillin left to center it between your eyes."

            # Vertical placement check
            if bottom_y >= hairline_y:
                return False, "Raise tefillin — it is below your hairline."

            return True, None  # Passed both checks

    return False, "Tefillin box detected but may not be centered or is too low."



#DECLERATIONS ======= NEED

def initialize(pic_path):
    global ref_ratio
    global reference_pose
    global ref_pic
    reference_pose = get_reference_pose(pic_path)

    ref_pic = cv2.imread(pic_path)

    ref_hairline = detect_hairline(ref_pic)
    ref_eyes = detect_eyes(ref_pic)

    ref_hairpoint = lowest_hairline_point(ref_eyes, ref_hairline)
    ref_features = face_features(ref_pic) #0 = chin, 1 = nose

    ref_bottom_dis = distance(ref_features[0][1:3], ref_features[1][1:3]) #distance from chin to nose
    ref_top_dis = distance(ref_features[1][1:3], ref_hairpoint) #distance from nose to forehead point
    ref_ratio = ratio(ref_bottom_dis, ref_top_dis)

initialize("/Users/koviressler/Desktop/DailyTefillin/people/kovi.JPG")

#TESTING ===========


cap = cv2.VideoCapture(0)  # Start webcam
pose_queue = deque(maxlen=3)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame

    # Draw pose and hairline guidance
    draw_hairline(frame)
    detect_face_and_paint(frame)

    cv2.putText(frame, "Press 'q' to evaluate (only if pose is aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Live Tefillin View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Run pose alignment check
        face_points = detect_face(frame)
        if not face_points:
            print("❌ Face not detected.")
            continue

        current_pose = detect_head_pose(frame, face_points)
        pose_queue.append(current_pose)
        smoothed_pose = smooth_poses(pose_queue)

        if smoothed_pose is None:
            print("❌ Pose not stable yet.")
            continue

        guidance = compare_poses(smoothed_pose, reference_pose)
        if guidance:
            print("⚠️ Adjust pose:", guidance)
            continue

        print("📸 Pose matched — evaluating tefillin placement...")
        result, feedback = evaluate_tefillin_placement(frame, "/Users/koviressler/Desktop/DailyTefillin/people/kovi.JPG")

        # Visual annotation logic:
        eye_coords = detect_eyes(frame)
        hairline_points = detect_hairline(frame)
        face_points = detect_face(frame)
        tefillin_results = detect_tefillin(frame)

        if eye_coords and eye_coords[0] and eye_coords[1]:
            # Draw center line between eyes
            left_x = eye_coords[0][1][1]
            right_x = eye_coords[1][0][1]
            mid_x = (left_x + right_x) // 2
            cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), (0, 255, 255), 2)

        if face_points:
            chin = face_points[152][1:3]
            nose = face_points[2][1:3]
            bottom_dis = distance(chin, nose)
            top_dis = bottom_dis * ref_ratio
            vector = np.array(nose) - np.array(chin)
            vector = vector / np.linalg.norm(vector)
            predicted_hairline = np.array(nose) + vector * top_dis
            hairline_y = int(predicted_hairline[1])
            cv2.line(frame, (0, hairline_y), (frame.shape[1], hairline_y), (255, 0, 0), 2)

        if tefillin_results:
            for result in tefillin_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, confidence in zip(boxes, confidences):
                    if confidence < 0.7:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if result else (0, 0, 255), 2)
                    cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if result else (0, 0, 255), 2)

        # Put final result message
        msg = "✅ Correct placement!" if result else f"❌ {feedback}"
        cv2.putText(frame, msg, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if result else (0, 0, 255), 2)

        cv2.imshow("Evaluation Result", frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Evaluation Result")


    elif key == ord('x'):
        print("Exiting.")
        break

cap.release()
cv2.destroyAllWindows()

exit()

# Usage - only for testing


#get_lowest_hairline_point(ref_pic)

#show_hairline_detected()


cap = cv2.VideoCapture(0)  # Open the default camera

pose_queue = deque(maxlen=1)  # For smoothing poses

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)

    #detect_tefillin(frame)
    #paint_tefillin(frame, detect_tefillin(frame))


    draw_hairline(frame)
    cv2.imshow('DailyTefillin', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()