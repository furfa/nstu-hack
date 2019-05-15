import cv2
import dlib
import threading
import time
import face_recognition
import pickle
from utils import BD
from roma_bd import BD_roman



OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data_roma.pickle', 'rb') as f:
     data = pickle.load(f)

known_face_names, known_face_encodings = data.get_data()

#We are not doing really face recognition
def doRecognizePerson(faceNames, fid, name):
    faceNames[fid] = name




def detectAndTrackMultipleFaces():
    #Open the first webcame device
    capture = cv2.VideoCapture('../test_out_04.avi')
    process_this_frame = True


    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

    try:
        while True:
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()

            #Resize the image to 320x240
            baseImage = cv2.resize(fullSizeBaseImage, (0,0), fx = 0.5, fy = 0.5)
            baseImage = baseImage[:, :, ::-1]

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(5)
            if pressedKey == ord('Q'):
                break



   
            resultImage = baseImage.copy()




            frameCounter += 1



            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )


                if trackingQuality < 7:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )





            if (frameCounter % 10) == 0:




                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)


                
                
                # y = top
                # x = left
                # h = bottom - y
                # w = right - x 



                
                
                # face_locations = face_recognition.face_locations(baseImage)
                face_locations = faceCascade.detectMultiScale(gray, 1.3, 5)
                # top, right, bottom, left

                

                fl = []
                # qwe = 1
                for (_x,_y,_w,_h) in face_locations:
                    if (_w**2 + _h**2)**0.5 < 100:
                        fl.append((_x,_y,_w,_h))
                face_locations = fl
                del fl

                face_locations = [(_y, _x+_w, _y+_h, _x) for (_x,_y,_w,_h) in face_locations]

                face_encodings = face_recognition.face_encodings(baseImage, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        
                        
                    face_names.append(name)

            




                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    y = top
                    x = left
                    h = bottom - y
                    w = right - x 




                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h



                    #Variable holding information which faceid we
                    #matched with
                    matchedFid = None

       
                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())
                        

                        

                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

      
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and
                             ( t_y <= y_bar   <= (t_y + t_h)) and
                             ( x   <= t_x_bar <= (x   + w  )) and
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid

                                        #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                  

                        print("Creating new tracker " + str(currentFaceID))

                        #Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x-10,
                                                            y-20,
                                                            x+w+10,
                                                            y+h+20))

                        faceTrackers[currentFaceID] = tracker

                        
                        faceNames[currentFaceID] = name

                        #Increase the currentFaceID counter
                        currentFaceID += 1





            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())


                top = t_y
                bottom = t_y + t_h
                right = t_x + t_w
                left = t_x


                
                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)
                print(faceNames)
                try:
                    cv2.putText(resultImage, faceNames[fid] , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                except KeyError:
                                cv2.putText(resultImage, 'Can`t rec' , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)



            resultImage = resultImage[:, :, ::-1]
            # largeResult = cv2.resize(resultImage,
            #                          (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            cv2.imshow("result-image", resultImage)







    except KeyboardInterrupt as e:
        pass

    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    detectAndTrackMultipleFaces()
