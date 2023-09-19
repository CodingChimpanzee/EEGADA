# EEGADA: EEG Automated Detector for Artifacts 

# Program explanation

  This program aims to detect the physiological artifacts via video information in real-time and marks those artifacts' source and their time in EEG data (e.g., gdf, edf, etc.). Information transition happens via TCP/IP connections. Also, current software can detect five movements, which are eye blinking, eye horizontal & vertical movement, jaw movement, and head movement.

<br>

# File list

* **movmement_class.py** : source code that describes how to detect the movements based on the MediaPipe landmarks.

* **movement_detection.py** : source code that activates artifact detection methodology and establish TCP/IP connections with EEG data.

* **DEMO.xml**: demo scenario.

* **README.md** & **README.txt**: manual with instructions.

<br>

# Program Requirements

## Operating system

* Windows 10, 11 (x86-64 bits)

## Software

* OpenViBE Designer, OpenViBE Acquisition Server (version 3.5.0)  <br>
link: http://openvibe.inria.fr/
* MediaPipe (version 0.10.0) <br>
link: https://developers.google.com/mediapipe
* PsychoPy (version 2023.2.0)  <br>
link: https://www.psychopy.org/
* Python version > 3.8, opencv-python (cv2), numpy packages are required.

<br>

# How to run a software 

**REMARK: You have to run the program in this order!**

1. Run OpenViBE acquisition server. If you want to just test the codes, please run the server in 'generic oscillator' mode.

2. Run DEMO.xml in OpenViBE designer. This is the OpenViBE scenario intended to run under 'generic oscillator' signals. If you want to run with real EEG signals, you have to use another scenario file in this step.

3. Run movement_detection.py, Then you may see the window to input the necessary information.
	
	  * Movement detection: you may input various movements in this section. 0 detects the overall described movements.
  
    * file name: you can save fps log file by the given name.
	
    * OpenViBE acquisition server IP address: IP address in OpenViBE acquisition server.
	
    * OpenViBE acquisition server port number: port number in OpenViBE acquisition server. 


4. If you input all those informations and wait for several seconds, then you may see the video window that detects your movement in real-time.

5. Press 'Esc' key to finish the movement detection. Then you may can see these files:

    * **DEMO.gdf** & **DEMO.edf** : EEG file with movement detection log.
    * **filename.txt** : fps log for every video input frame. This will be saved with user input  file name.
    
    *NOTE: you have to set the file save path in DEMO.xml and movement_detection.py (at very last line)*
<br>

# EEG file description

  You can see the event markers that recorded the detected movements from this movement detection. These movements are described as (Note: left/right is participant's left/right):
	
* Eye blinking: 11 (eye close), 111 (eye open)

* Left eye movement: 21 (move), 121 (stop)
	
* Right eye movement: 22 (move), 122 (stop)

* Jaw movement: 41 (move), 141 (stop)

* Head yaw movement: 51 (move), 151 (stop)

* Head pitch movement: 61 (move), 161 (stop)

* Head roll movement: 71 (move), 171 (stop)

<br>

# TroubleShoot

1. If the TCP/IP connection is not working, please shut down the program and run it again. This usually happens when you run the program in wrong order.

2. If there are multiple movement observations for eye movement, this is intended. We observed physiological artifacts in EEG for relatively small eye movements, thus we made eye movement detection more sensitive.

<br>

# Related papers
* S. Kang, K. Won, H. Kim, J. Baek, M. Ahn, and S.C. Jun, “Achieving effective artifact subspace reconstruction in EEG using real-time video-based artifact identification”, IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2023, Oct. 2023. (accepted, oral presentation)

<br>

# Copyrights
This software is registered with the Korea Copyright Commission (C-2023-041615). This software can be utilized for academic purposes.

<br>

# References
* C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C. L. Chang, M. Yong, J. Lee, W.T. Chang, W. Hua, M. Georg, and M. Grundmann, “Mediapipe: A framework for perceiving and processing reality”, Third Workshop on Computer Vision for AR/VR at IEEE Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, vol. 2019.
* Y. Renard, F. Lotte, G. Gibert, M. Congedo, E. Maby, V. Delannoy, O. Bertrand, and A. Lécuyer, “Openvibe: An open-source software platform to design, test, and use brain–computer interfaces in real and virtual environments”, Presence, vol. 19, no. 1, pp. 35-53, 2010.
* T. Soukupova and J. Cech, “Eye blink detection using facial landmarks”, 21st computer vision winter workshop, Rimske Toplice, Slovenia, 2016, p. 2.
* R. Chinthala, S. Katkoori, C. S. Rodriguez, and M. J. Mifsud, "An Internet of Medical Things (IoMT) Approach for Remote Assessment of Head and Neck Cancer Patients", 2022 IEEE International Symposium on Smart Electronic Systems (iSES), Warangal, India, 2022, pp. 124-129.
* A. M. Al-Nuimi and G. J. Mohammed, "Face Direction Estimation based on Mediapipe Landmarks," 2021 7th International Conference on Contemporary Information Technology and Mathematics (ICCITM), Mosul, Iraq, 2021, pp. 185-190.

<br>

If you have any inquires, please contact: kanghyun51015@gm.gist.ac.kr
