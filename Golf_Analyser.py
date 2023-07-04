import sys
import cv2
import os
import shutil
import mediapipe as mp
import numpy as np
from firebase_manager import FireBaseManager
from Golf_Analyser import *
from sklearn import cluster

def detectionModel():
    # sets up the pose detection model 

    # 1) mp_drawing
    # 2) mp_pose
    # return both of those
    mp_drawing  = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    return mp_drawing, mp_pose

def pose_process_image(image,pose):
    
    # do that, store the results after pose processing it
    # return original image , results
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_info = pose.process(rgb)
    return image, processed_info

def calculate_angle(p1,p2,p3):

    first = np.array(p1)  # information about the first joint 
    second = np.array(p2)
    third = np.array(p3)

    # first[0] <-- x position
    # first[1] <-- y position

    a = np.arctan2((third[1]- second[1]),(third[0]- second[0]))
    b = np.arctan2((first[1]- second[1]),(first[0]- second[0]))

    radians = a - b

    # radian = np.arctan((third[1]- second[1]),(third[0]- second[0])) - np.arctan((first[1]- second[1]),(first[0]- second[0]))
    degree = np.abs(radians * 180 / np.pi)

    if degree > 180.0:
        degree = 360 - degree

    return round(degree)

def plot_angle(p1,p2,p3, landmarks, image):

    # calculate the angle

    # landmarks[p1].x
    a = [landmarks[p1].x,landmarks[p1].y]
    
    b = [landmarks[p2].x,landmarks[p2].y]

    c = [landmarks[p3].x,landmarks[p3].y]


    angle = calculate_angle(a,b,c)

    x = b[1] * image.shape[1] 
    y = b[0] * image.shape[0]
    org = tuple(x,y)
    
    imageWithAngleText = draw_angle(org,image,angle) 
    return imageWithAngleText

def plot_angle(a1, a2, b1, b2, c1, c2, image):



    # calculate the angle


    a = [a1, a2]
    b = [b1, b2]
    c = [c1, c2]


    angle = calculate_angle(a,b,c)

    print(angle)

    x = b[1] * image.shape[1] 
    y = b[0] * image.shape[0]
    org = tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int))
    
    imageWithAngleText = draw_angle(org,image,angle) 
    return imageWithAngleText

def draw_angle(org : tuple, image, angle):

    color = (0,255,0)
    imageWithText = cv2.putText(image,str(angle),org,cv2.FONT_HERSHEY_COMPLEX, 1,color,2)
    return imageWithText

def draw_image_angle_elbow(frame, landmarkResults):


    # Gather body part information
    shoulder = [landmarkResults.pose_landmarks.landmark[11].x, landmarkResults.pose_landmarks.landmark[11].y]
    elbow = [landmarkResults.pose_landmarks.landmark[13].x, landmarkResults.pose_landmarks.landmark[13].y]
    wrist = [landmarkResults.pose_landmarks.landmark[15].x, landmarkResults.pose_landmarks.landmark[15].y]

    # Calculate the angle
    angle = calculate_angle(shoulder, elbow, wrist)

    posX = int(elbow[0] * frame.shape[1])
    posY = int(elbow[1] * frame.shape[0])
    position = (posX, posY)

    # Draw the angle
    cv2.putText(frame, str(angle), position, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255),2)

def draw_joint(frame, landmarkResults, bp1, bp2, bp3):

    # Gather body part information
    bodypart1 = [landmarkResults.pose_landmarks.landmark[bp1].x, landmarkResults.pose_landmarks.landmark[bp1].y]
    bodypart2 = [landmarkResults.pose_landmarks.landmark[bp2].x, landmarkResults.pose_landmarks.landmark[bp2].y]
    bodypart3 = [landmarkResults.pose_landmarks.landmark[bp3].x, landmarkResults.pose_landmarks.landmark[bp3].y]

    # Calculate the angle
    angle = calculate_angle(bodypart1, bodypart2, bodypart3)

    posX = int(bodypart2[0] * frame.shape[1])
    posY = int(bodypart2[1] * frame.shape[0])
    position = (posX, posY)

    # Draw the angle
    cv2.putText(frame, str(angle), position, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255),2)

    return angle

def getLandMarks(frame, pose):
    # Input: Frame that needs to be analyzed by mediapipe (BGR format)
    # Output: information about the landmarks of that images
    
    # Covert frame to RGB format
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgbFrame)
    return results 

def drawLandMarks(landmarks, frame, mp_drawing, mp_pose):
    mp_drawing.draw_landmarks(frame, landmarks.pose_landmarks,mp_pose.POSE_CONNECTIONS)


    jointangle = []
    jointangle.append(draw_joint(frame,landmarks,12,14,16)) # rightelbows
    jointangle.append(draw_joint(frame,landmarks,11,13,15)) # Leftelbows
    jointangle.append(draw_joint(frame,landmarks,14,12,24)) # rightshoulders
    jointangle.append(draw_joint(frame,landmarks,13,11,23)) # leftshoulders
    jointangle.append(draw_joint(frame,landmarks,12,24,26)) # righthips
    jointangle.append(draw_joint(frame,landmarks,11,23,25)) # lefthips

    return jointangle

def getvideodata(video_name,mp_drawing,mp_pose, pose):
    # capObject is the actual video
    capObject = cv2.VideoCapture(video_name)

    if(capObject.isOpened() == False):
        print("Could not print video")

    # create a while loop that stays true as long as capObject is opened4  
    allframeangles = []
    while(capObject.isOpened()):

        success,videoFrame = capObject.read()
        # success - whether there is frame left
        # videoFrame - the actual frame
        if(not success):
            break


        result = getLandMarks(videoFrame,pose)
        jointangle = drawLandMarks(result,videoFrame,mp_drawing,mp_pose)
        allframeangles.append(jointangle)
    return allframeangles

def get_index(labels, prediction):
    indexarr = []
    for i in range (len(labels)):
        if(labels[i] == prediction):
            indexarr.append(i)
    return indexarr

def display_frames(video, indexarr):
    # capObject is the actual video
    capObject = cv2.VideoCapture(video)

    if(capObject.isOpened() == False):
        print("Could not print video")

    index = 0
    while(capObject.isOpened()):
        success,videoFrame = capObject.read()
        # success - whether there is frame left
        # videoFrame - the actual frame
        if(not success):
            break

        if(index in indexarr):
            # show the image
            cv2.imshow('Image', videoFrame)

            if cv2.waitKey(0) == ord('q'):
                break
        index += 1

def get_start_end_clusters(labels): 

    cluster_info = []
    startIndex = 0
    for i in range(len(labels)-1):
        if(labels[i+1] != labels[i]):
            cluster_info.append({'label': labels[i], 'start': startIndex, 'end': i})
            startIndex = i+1
    
    cluster_info.append({'label': labels[len(labels)-1], 'start': startIndex, 'end': len(labels)-1})
    return cluster_info

def get_closest_neighbor(selected_frame_angle, indexes_other_video, angles_other_video):
    a = np.array(selected_frame_angle)
    
    # Go through entire cluster and find frame with the min distance
    mindistance = sys.maxsize
    index_other_frame = indexes_other_video[0]


    for index in indexes_other_video:
        cluster_angles = angles_other_video[index]
        b = np.array(cluster_angles)
        distance = np.linalg.norm(a - b)

        # found frame in the cluster with smaller distance
        if(distance < mindistance):
            mindistance = distance
            index_other_frame = index

    return index_other_frame, distance
    
def compare_video(cluster_info, video1_angles, video2_angles, video2_model):
    index_coach_frames = []
    index_student_frames = []

    for cluster in cluster_info:
        # Pick a frame in each cluster
        selected_vid1_frame = ((cluster['end'] + cluster['start'])) // 2

        # Catagorize with cluster the frame will belong in the other video
        prediction = video2_model.predict([video1_angles[selected_vid1_frame]])

        # Get all the frames that belong to the cluster of prediction
        vid2_cluster_index = get_index(video2_model.labels_, prediction[0])

        # Find the closest frame in the cluster to the selected frame 
        selected_vid2_frame, quality = get_closest_neighbor(video1_angles[selected_vid1_frame], vid2_cluster_index, video2_angles)

        # # Check if selected frames between coach and student pass a standard of quality based on distance
        # quality_requirement = 65
        # if(quality < quality_requirement or True):

        # Store index of selected coach frame
        index_coach_frames.append(selected_vid1_frame)
        # Store index of selected student frame
        index_student_frames.append(selected_vid2_frame)

    return index_coach_frames, index_student_frames

def retrieve_frames(video_name, indexarr):

    # image array to be returned
    image_array = [None] * len(indexarr)

    # capObject is the actual video
    capObject = cv2.VideoCapture(video_name)

    if(capObject.isOpened() == False):
        print("Could not print video")

    index = 0
    while(capObject.isOpened()):
        success,videoFrame = capObject.read()
        # success - whether there is frame left
        # videoFrame - the actual frame
        if(not success):
            break

        if(index in indexarr):

            for i in range(len(indexarr)):
                if indexarr[i] == index:
                    # insert the image into the array
                    image_array[i] = videoFrame
        index += 1

    capObject.release()

    return image_array

def video_analyzer(video_1, video_2):

    # MediaPipe Set up
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Get video data
    coach_angles = getvideodata(video_1,mp_drawing,mp_pose,pose)
    student_angles = getvideodata(video_2,mp_drawing,mp_pose,pose)

    # Create machine learning object for coach (model)
    coachmodel = cluster.KMeans(n_clusters=6, random_state=0, n_init=10)
    coachmodel.fit(coach_angles)

    # Create machine learning object for student (model)
    studentmodel = cluster.KMeans(n_clusters=6, random_state=0, n_init=10)
    studentmodel.fit(student_angles)

    cluster_coach_info = get_start_end_clusters(coachmodel.labels_)


    # Array of indexes of frames that are similar to each other
    index_coach_frames, index_student_frames = compare_video(cluster_coach_info,coach_angles,student_angles,studentmodel)

    # Use the indexes above to find the frames we need to upload from the video
    # Array of images/frames that are similar to each other
    coach_frames = retrieve_frames(video_1,index_coach_frames)
    student_frames = retrieve_frames(video_2,index_student_frames)

    # Check if folders exist, if not then create folders
    if not os.path.isdir('coach_frames_folder'):
        os.mkdir('coach_frames_folder')
    if not os.path.isdir('student_frames_folder'):
        os.mkdir('student_frames_folder')

    # upload images in arrays to folders in local computer
    for i in range(len(coach_frames)):
        cv2.imwrite(f'coach_frames_folder/coach_image_{i}.jpg', coach_frames[i])
        cv2.imwrite(f'student_frames_folder/student_image_{i}.jpg', student_frames[i])


    try:
        print("trying to access firebase storage")
        # upload images from local computer to firebase
        fb_manager = FireBaseManager()
        public_urls = []
        for i in range(len(coach_frames)):
            public_urls.append((fb_manager.upload_file(f'Coachframes/coach_image_{i}.jpg', f'coach_frames_folder/coach_image_{i}.jpg'),fb_manager.upload_file(f'Studentframes/student_image_{i}.jpg', f'student_frames_folder/student_image_{i}.jpg')))
        print("successfully accessed and uploaded firebase storage files")
    except Exception as e:
        print(e)


    # delete files and folders after they have been stored online
    if os.path.isdir('coach_frames_folder'):
        shutil.rmtree('coach_frames_folder')
    if os.path.isdir('student_frames_folder'):
        shutil.rmtree('student_frames_folder')

    return public_urls