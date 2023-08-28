# Original code form the official MediaPipe documentation
# https://google.github.io/mediapipe/solutions/face_mesh.html
# Other parts were written by Sunghyun Kang (kanghyun51015@gm.gist.ac.kr)
# Note: left/right position is based on the view of original camera

# dependencies: cv2 (opencv-python), mediapipe, numpy
import cv2
import socket
import mediapipe as mp
import numpy as np
import os
import time
from psychopy import gui

# import movement class
from movement_class import EAR, iris_movement, jaw_movement, yaw_movement, pitch_movement, roll_movement

# Face mesh declaration
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# FPS Counter (For evaluation)
FPS_counter = []

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

# sending type
# transform a value into an array of byte values in little-endian order.
def to_byte(value, length):
    for _ in range(length):
        yield value%256
        value//=256

def mediapipe_running(client_socket, padding, timestamp, stimulation, experiment_no, filename):

  # compare previous and next EAR value
  prev_ear_left = ''
  prev_ear_right = ''

  # compare previous and next iris location
  prev_iris_left_move = ''
  prev_iris_left = 0
  prev_iris_right_move = ''
  prev_iris_right = 0

  # compare previous and next yaw movement
  prev_yaw_move = ''
  yaw_move_detect = ''
  prev_yaw = 0
  
  # compare previous and next pitch movement
  prev_pit_move = ''
  pit_move_detect = ''
  prev_pit = 0
  
  # compare previous and next roll movement
  prev_roll_move = ''
  roll_move_detect = ''
  prev_roll = 0

  # jaw move
  prev_jaw_move = ''

  # For video input, draw the specific face area that we will detect the movement
  # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

  # If you want to put a video input, use this code (must include path)
  # cap = cv2.VideoCapture('video that you want to play')
  # cap.set(cv2.CAP_PROP_POS_MSEC, 100)

  # Webcam input (must have camera)
  cap = cv2.VideoCapture(0)

  # error handling (If there is no camera)
  if not cap.isOpened():
    print('NO CAMERA!')
    return -1
  
  # If you want to save the video, use this code
  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  # fps = cap.get(cv2.CAP_PROP_FPS)
  # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  # out = cv2.VideoWriter(BASE_DIR + '/' + filename + '.mp4', fourcc, fps, (int(width), int(height)))

  # previous time
  prev = 0.0

  # if the signal received properly, then do the face mesh recording
  with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
      
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          # continue
          break
        # saving part
        elapsed_time = time.time() - prev
        prev = elapsed_time + prev
        cur_fps = round(1/elapsed_time, 2)
        FPS_counter.append(cur_fps)

        # write the image by the current fps
        # out.write(image)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # This works per frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # To get the current timestemp
        # current = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # If the results for multi facial landmarks exists => check all points!
        if results.multi_face_landmarks:
          # Depict all landmakrs
          for face_landmarks in results.multi_face_landmarks:
            
            # prev_temporal, temporal_detect = temp_front(face_landmarks, Temporalis, prev_temporal, 'Temporalis', current)      
            # prev_frontal, frontal_detect = temp_front(face_landmarks, Frontalis, prev_frontal, 'Frontalis', current)
            if 0 in experiment_no:
              prev_ear_left, ear_left_detect = EAR(Left_eye, face_landmarks, prev_ear_left)
              prev_ear_right, ear_right_detect = EAR(Right_eye, face_landmarks, prev_ear_right)
              prev_iris_left, prev_iris_left_move, iris_left_detect = iris_movement(Left_iris, face_landmarks, prev_iris_left, prev_iris_left_move)
              prev_iris_right, prev_iris_right_move, iris_right_detect = iris_movement(Right_iris, face_landmarks, prev_iris_right, prev_iris_right_move)
              prev_jaw_move, jaw_move_detect = jaw_movement(face_landmarks, prev_jaw_move)
            if 1 in experiment_no:
              prev_ear_left, ear_left_detect = EAR(Left_eye, face_landmarks, prev_ear_left)
              prev_ear_right, ear_right_detect = EAR(Right_eye, face_landmarks, prev_ear_right)
            if (2 in experiment_no) or (3 in experiment_no):
              prev_iris_left, prev_iris_left_move, iris_left_detect = iris_movement(Left_iris, face_landmarks, prev_iris_left, prev_iris_left_move)
              prev_iris_right, prev_iris_right_move, iris_right_detect = iris_movement(Right_iris, face_landmarks, prev_iris_right, prev_iris_right_move)
            if 4 in experiment_no:
              prev_jaw_move, jaw_move_detect = jaw_movement(face_landmarks, prev_jaw_move)
            if 5 in experiment_no:
              prev_yaw, prev_yaw_move, yaw_move_detect = yaw_movement(face_landmarks, prev_yaw, prev_yaw_move)
              prev_pit, prev_pit_move, pit_move_detect = pitch_movement(face_landmarks, prev_pit, prev_pit_move)
              prev_roll, prev_roll_move, roll_move_detect = roll_movement(face_landmarks, prev_roll, prev_roll_move)
            
            
            # Draw all face landmarks
            # This part significantly impacts the fps, thus it is not recommended.
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list= face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list= face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None, 
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())

        else:
          # Show the image
          cv2.imshow('experiment_show', image)
          # End signal
          if cv2.waitKey(27) & 0xFF == 27:
            break
          continue
        
        # Show the fps
        # cv2.putText(image, str(round(cap.get(cv2.CAP_PROP_FPS), 2))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        # Show the image
        cv2.imshow('experiment_show', image)

        # print out the movements
        # which can be understood as sending signals to tcp connection
        # first digit: event start(0)/end(1)
        # second digit: which event (1: eye close/open, 2: iris movement)
        # third digit: which location (1: left, 2: right)
        sending_events = []

        # DEMO: all movement detection
        if 0 in experiment_no:
          if (ear_left_detect == 'close') or (ear_right_detect == 'close'):
            sending_events.append(11)
          elif (ear_left_detect == 'open') or (ear_right_detect == 'open'):
            sending_events.append(111)

          if iris_left_detect == 'move':
            sending_events.append(21)
          elif iris_left_detect == 'stop':
            sending_events.append(121)
          if iris_right_detect == 'move':
            sending_events.append(22)
          elif iris_right_detect == 'stop':
            sending_events.append(122)

          if jaw_move_detect == 'open':
            sending_events.append(41)
          elif jaw_move_detect == 'close':
            sending_events.append(141)

          if yaw_move_detect == 'move':
            sending_events.append(51)
          elif yaw_move_detect == 'stop':
            sending_events.append(151)
          
          if pit_move_detect == 'move':
            sending_events.append(61)
          elif pit_move_detect == 'stop':
            sending_events.append(161)

          if roll_move_detect == 'move':
            sending_events.append(71)
          elif roll_move_detect == 'stop':
            sending_events.append(171)

        # eye detection
        if 1 in experiment_no:
          if (ear_left_detect == 'close') or (ear_right_detect == 'close'):
            sending_events.append(11)
          elif (ear_left_detect == 'open') or (ear_right_detect == 'open'):
            sending_events.append(111)
        
        # iris detection
        if (2 in experiment_no) or (3 in experiment_no):
          if iris_left_detect == 'move':
            sending_events.append(21)
          elif iris_left_detect == 'stop':
            sending_events.append(121)
          if iris_right_detect == 'move':
            sending_events.append(22)
          elif iris_right_detect == 'stop':
            sending_events.append(122)

        # jaw clenching detection
        # This may include jaw biting
        if 4 in experiment_no:
          if jaw_move_detect == 'open':
            sending_events.append(41)
          elif jaw_move_detect == 'close':
            sending_events.append(141)

        if 5 in experiment_no:
          if yaw_move_detect == 'move':
            sending_events.append(51)
          elif yaw_move_detect == 'stop':
            sending_events.append(151)
          
          if pit_move_detect == 'move':
            sending_events.append(61)
          elif pit_move_detect == 'stop':
            sending_events.append(161)

          if roll_move_detect == 'move':
            sending_events.append(71)
          elif roll_move_detect == 'stop':
            sending_events.append(171)
        
        # send the information using tcp connection
        if len(sending_events) > 0:
          client_socket.sendall(bytes(padding + stimulation + timestamp))
          for id in sending_events:
            event_id = list(to_byte(int(id), 8))
            client_socket.sendall(bytes(padding + event_id + timestamp))

        # End signal
        if cv2.waitKey(27) & 0xFF == 27:
          break

  cap.release()
  # out.release()
  cv2.destroyAllWindows()


# Program init
if __name__ == '__main__':
  
  # input for the experiment settings
  expInfo = {
    'movement detection (0: demo, 1: eye blinking, 2: eye horizontal movememnt, 3: eye vertical movement, 4: jaw movement, 5: head movement)': f"1,4",
    'file name': '001',
    'OpenViBE acquisition server IP address': 'localhost',
    'OpenViBE acquisition server port number': '15361'
  }
  # GUI setting
  dlg = gui.DlgFromDict(sortKeys=False, dictionary=expInfo, title='experiment_input')
  if dlg.OK == False:
    exit() # user pressed cancel
  

  # input client_address, port number
  # initiate the tcp connection
  # internal connection to OpenViBE
  client_address = expInfo['OpenViBE acquisition server IP address']
  port = int(expInfo['OpenViBE acquisition server port number'])

  experiment_no_input = expInfo['movement detection (0: demo, 1: eye blinking, 2: eye horizontal movememnt, 3: eye vertical movement, 4: jaw movement, 5: head movement)']
  filename = expInfo['file name']

  # make experiment number
  experiment_no_list = experiment_no_input.split(',')
  experiment_no = []
  for i in experiment_no_list:
    experiment_no.append(int(i))

  # TCP socket creation
  try:
      client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      print("Client socket creation success!")
      print(client_address, port)
      client_socket.settimeout(None)
  except Exception as e:
      print(f'error message : {e}')
      exit()

  # TCP connection (acquisition server)
  try:
      client_socket.connect((client_address, port))
      print("Socket connection success!")
  except Exception as e:
      print(f'error message : {e}')
      exit()


  # Declare padding and timestamp
  padding = list(to_byte(4, 8))
  timestamp = [0]*8
  # Declare stimulus id
  stimulation = list(to_byte(51, 8))

  # mediapipe will send the signal using TCP/IP communication
  mediapipe_running(client_socket, padding, timestamp, stimulation, experiment_no, filename)
  
  # end connection
  client_socket.close()
  
  np.savetxt(filename + '.txt', FPS_counter, delimiter=',')