# This part is written by Sunghyun Kang (kanghyun51015@gm.gist.ac.kr)
# Note: left/right position is based on the view of original camera

# dependencies: numpy
import numpy as np

# coordinates list
# Temporalis and Frontalis muscle coordinates
Temporalis = [54, 68, 21, 71, 70, 162, 139, 156, 124, 35, 127, 34, 143, 234, 227, 116]
Temporalis = Temporalis + [284, 298, 251, 301, 300, 389, 368, 383, 353, 356, 264, 372, 265, 454, 447, 345]
Frontalis = [67, 69, 103, 104, 108, 109]
Frontalis = Frontalis + [338, 337, 297, 299, 332, 333]

# left and right eye coordinates
Right_eye = [33, 160, 158, 133, 153, 144]
Left_eye = [362, 385, 387, 263, 373, 380]

# iris location coordinates
# 468, 469, 470, 471, 472
# 473, 474, 475, 476, 477
Right_iris = [133, 159, 33, 145, 472]
Left_iris = [263, 386, 362, 374, 477]

# eye blinking calculation
# Utilized eye aspect ratio (EAR) equation!
# Citation: "Eye blink detection using facial landmarks"
# right eye: 33, 160, 158, 133, 153, 144
# left eye: 362, 385, 387, 263, 373, 380
def EAR(eyelist, face_landmarks, prev_ear):

  # initialize return value
  p1x = face_landmarks.landmark[eyelist[0]].x
  p1y = face_landmarks.landmark[eyelist[0]].y
  p2x = face_landmarks.landmark[eyelist[1]].x
  p2y = face_landmarks.landmark[eyelist[1]].y
  p3x = face_landmarks.landmark[eyelist[2]].x
  p3y = face_landmarks.landmark[eyelist[2]].y
  p4x = face_landmarks.landmark[eyelist[3]].x
  p4y = face_landmarks.landmark[eyelist[3]].y
  p5x = face_landmarks.landmark[eyelist[4]].x
  p5y = face_landmarks.landmark[eyelist[4]].y
  p6x = face_landmarks.landmark[eyelist[5]].x
  p6y = face_landmarks.landmark[eyelist[5]].y

  temp1 = np.square(p2x-p6x)
  temp2= np.square(p2y-p6y)

  temp3 = np.square(p3x-p5x)
  temp4 = np.square(p3y-p5y)

  temp5 = np.square(p1x-p4x)
  temp6 = np.square(p1y-p4y)

  ear = (np.sqrt(temp1 + temp2) + np.sqrt(temp3 + temp4))/np.sqrt(temp5 + temp6)

  # ear threshold
  # open -> close: 0.22
  # close -> open: 0.3

  if len(prev_ear) == 0:
    return 'open', None
  
  if prev_ear == 'open' and ear < 0.22:
    return 'close', 'close'
  
  elif prev_ear == 'close' and ear > 0.3:
    return 'open', 'open'
  
  else:
    return prev_ear, None


# Iris movement detection (Nor for current use)
def iris_movement(iris_list, face_landmarks, prev_iris, prev_move):

  east_x = face_landmarks.landmark[iris_list[0]].x
  east_y = face_landmarks.landmark[iris_list[0]].y
  north_x = face_landmarks.landmark[iris_list[1]].x
  north_y = face_landmarks.landmark[iris_list[1]].y
  west_x = face_landmarks.landmark[iris_list[2]].x
  west_y = face_landmarks.landmark[iris_list[2]].y
  south_x = face_landmarks.landmark[iris_list[3]].x
  south_y = face_landmarks.landmark[iris_list[3]].y

  origin_x = ((east_x + west_x)/2 + (north_x + south_x)/2)/2
  origin_y = ((east_y + west_y)/2 + (north_y + south_y)/2)/2

  iris_x = face_landmarks.landmark[iris_list[4]].x
  iris_y = face_landmarks.landmark[iris_list[4]].y

  distance = np.sqrt((iris_x - origin_x) ** 2+(iris_y - origin_y) **2)

  if len(prev_move) == 0:
    return distance, 'stop', None
  
  if abs(distance - prev_iris) > 0.0005 and prev_move == 'stop':
    return distance, 'move', 'move'
  
  if abs(distance - prev_iris) < 0.0005 and prev_move == 'move':
    return distance, 'stop', 'stop'

  return distance, prev_move, None

# jaw movement recognition
# reference: "An Internet of Medical Things (IoMT) Approach for Remote Assessment of Head and Neck Cancer Patients"
# nose: 1
# chin: 199
# upper lip: 13
# lower lip: 14
def jaw_movement(face_landmarks, prev_ratio):

  # get the coordinates
  nose_x = face_landmarks.landmark[1].x
  nose_y = face_landmarks.landmark[1].y

  chin_x = face_landmarks.landmark[199].x
  chin_y = face_landmarks.landmark[199].y

  upper_lip_x = face_landmarks.landmark[13].x
  upper_lip_y = face_landmarks.landmark[13].y

  lower_lip_x = face_landmarks.landmark[14].x
  lower_lip_y = face_landmarks.landmark[14].y

  nose_chin_distance = np.sqrt((nose_x - chin_x) ** 2+(nose_y - chin_y) **2)
  mouth_distance = np.sqrt((upper_lip_x - lower_lip_x) ** 2+(upper_lip_y - lower_lip_y) **2)

  # if the ratio goes to zero, then it is jaw close
  ratio = mouth_distance / nose_chin_distance

  if ratio >= 0.05 and prev_ratio == 'close':
    return 'open', 'open'

  if ratio < 0.05 and prev_ratio == 'open':
    return 'close', 'close'
  
  if len(prev_ratio) == 0:
    if ratio > 0.05:
      return 'open', None
    else:
      return 'close', None

  return prev_ratio, None


# facial angle (head pose) recognition
# Reference: "Face Direction Estimation based on Mediapipe Landmarks"
# Yaw angle coordinates: 18, 50, 180
def yaw_movement(face_landmarks, prev_yaw, prev_yaw_mv):

  # coordinates
  x_18 = face_landmarks.landmark[18].x
  y_18 = face_landmarks.landmark[18].y
  x_50 = face_landmarks.landmark[50].x
  y_50 = face_landmarks.landmark[50].y
  x_280 = face_landmarks.landmark[280].x
  y_280 = face_landmarks.landmark[280].y

  # calculate the yaw angle
  y_a = np.sqrt((x_50 - x_18)**2 + (y_50 - y_18)**2)
  y_b = np.sqrt((x_280 - x_18)**2 + (y_280 - y_18)**2)

  y_v1 = [x_280-x_50, y_280-y_50]
  y_v2 = [x_18-x_50, y_18-y_50]
  y_ver_dist4 = np.cross(y_v1, y_v2)/np.linalg.norm(y_v1)

  y_r = np.sqrt(y_a**2 - y_ver_dist4**2)
  y_l = np.sqrt(y_b**2 - y_ver_dist4**2)

  if y_a >= y_b and y_r/y_l < 2:
      theta_yaw = np.arcsin(1-(y_r/y_l))
  elif y_a < y_b and y_l/y_r < 2:
      theta_yaw = np.arcsin(1-(y_l/y_r))
  else:
    theta_yaw = -10000

  if prev_yaw == 10000:
    return theta_yaw, 'stop', None

  if abs(theta_yaw - prev_yaw) >= 0.1 and prev_yaw_mv == 'stop':
    return theta_yaw, 'move', 'move'

  if abs(theta_yaw - prev_yaw) <= 0.015 and prev_yaw_mv == 'move':
    return theta_yaw, 'stop', 'stop'

  return theta_yaw, prev_yaw_mv, None


# Pitch angle coordinates: 4, 50, 280
def pitch_movement(face_landmarks, prev_pit, prev_pit_mv):

  # coordinates
  x_4 = face_landmarks.landmark[4].x
  y_4 = face_landmarks.landmark[4].y
  x_50 = face_landmarks.landmark[50].x
  y_50 = face_landmarks.landmark[50].y
  x_280 = face_landmarks.landmark[280].x
  y_280 = face_landmarks.landmark[280].y

  # calculate the yaw angle
  y_a = np.sqrt((x_50 - x_4)**2 + (y_50 - y_4)**2)
  y_b = np.sqrt((x_280 - x_4)**2 + (y_280 - y_4)**2)

  y_v1 = [x_280-x_50, y_280-y_50]
  y_v2 = [x_4-x_50, y_4-y_50]
  y_ver_dist4 = np.cross(y_v1, y_v2)/np.linalg.norm(y_v1)

  if y_b > y_ver_dist4 and np.abs(y_ver_dist4/y_b) < 1:
      theta_pit_left = np.arcsin(y_ver_dist4/y_b)
  elif y_b < y_ver_dist4 and np.abs(y_b/y_ver_dist4) < 1:
     theta_pit_left = np.arcsin(y_b/y_ver_dist4)
  else:
    theta_pit_left = 0

  if y_a > y_ver_dist4 and np.abs(y_ver_dist4/y_a) < 1:
      theta_pit_right = np.arcsin(y_ver_dist4/y_a)
  elif y_a < y_ver_dist4 and np.abs(y_a/y_ver_dist4) < 1:
     theta_pit_right = np.arcsin(y_a/y_ver_dist4)
  else:
    theta_pit_right = 0

  theta_pit = (theta_pit_left+theta_pit_right)/2

  if prev_pit == 0:
    return theta_pit, 'stop', None

  if abs(theta_pit - prev_pit) >= 0.1 and prev_pit_mv == 'stop':
    return theta_pit, 'move', 'move'

  if abs(theta_pit - prev_pit) <= 0.015 and prev_pit_mv == 'move':
    return theta_pit, 'stop', 'stop'

  return theta_pit, prev_pit_mv, None

# Roll angle coordinates: 50, 280
def roll_movement(face_landmarks, prev_roll, prev_roll_mv):

  # coordinates
  x_50 = face_landmarks.landmark[50].x
  y_50 = face_landmarks.landmark[50].y
  x_280 = face_landmarks.landmark[280].x
  y_280 = face_landmarks.landmark[280].y

  # calculate the yaw angle
  y_a = np.sqrt((x_280 - x_50)**2 + (y_280 - y_50)**2)
  y_b = np.sqrt((y_280 - y_50)**2)

  if y_a > y_b and np.abs(y_b/y_a) < 1:
      theta_roll = np.arcsin(y_b/y_a)
  elif y_a < y_b and np.abs(y_a/y_b) < 1:
      theta_roll = np.arcsin(y_a/y_b)
  else:
    theta_roll = 0
  
  if prev_roll == 0:
    return theta_roll, 'stop', None

  if abs(theta_roll - prev_roll) >= 0.1 and prev_roll_mv == 'stop':
    return theta_roll, 'move', 'move'

  if abs(theta_roll - prev_roll) <= 0.015 and prev_roll_mv == 'move':
    return theta_roll, 'stop', 'stop'

  return theta_roll, prev_roll_mv, None