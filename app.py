from flask import Flask, render_template, jsonify, request

import test

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])

def web_page():
    if request.method=='POST':
        
        
        import cv2
        import numpy as np
        import mediapipe as mp
        import csv
        import pandas as pd
        import pyttsx3
        import joblib


        #  Annotations

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils


        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = model.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results



        def draw_styled_landmarks_G(image, results):
            mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(0,128,0), thickness = 2,circle_radius=1)
                                    )
            mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(0,128,0), thickness = 2,circle_radius=1)
                                    )


        def draw_styled_landmarks_np_nf_B(image, results):
            mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=5),
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness = 2,circle_radius=2)
                                    )
            mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=5),
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness = 2,circle_radius=2)
                                    )


        # Speech integration

        def speak(text):
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', 150)

            #Setting the voice (male or female)
            #male ID = 0
            #female ID = 1
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)

            # Taking Text input
            engine.say(text)
            engine.runAndWait()


        # Functions for sign Detections


        model_L = joblib.load('MP_model_head.pkl')



        def sign_output(sign_list, sentence, sentence_out):
            with open('multi_sign.csv') as multisign_file:
                sign_list = csv.reader(multisign_file)
                for row in sign_list:
                    if sentence[-1] == row[-1]:
                        if sentence[-2] == row[-2]:
                            sentence_out.append(row[0])
                            break
                    else:
                        continue



        def detect(vidsource):
            
            sentence = []
            sentence_out = []
            
            predictions = []
            
            last_sign_list = []
            one_sign_list = []
            
            #Setting probability threshold
            threshold = 0.9
            
            #set 3 predictions per sign as the minimum number for confirmation of probability
            pr = 3
            
            #Load complex signs from multisign file
            with open('multi_sign.csv') as multisign_file:
                sign_list = csv.reader(multisign_file)
                for row in sign_list:
                    last_sign_list.append(row[-1])
            
            #Loading single signs from singlesign file
            with open('single_sign.csv') as singlesign_file:
                singlesign_list = csv.reader(singlesign_file)
                for row in singlesign_list:
                    one_sign_list.append(row[0])
            
            #Source of video feed detection
            cap = cv2.VideoCapture(vidsource)
            
            #Estasblish the mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                
                while cap.isOpened():

                    #Get frame readings
                    ret, frame = cap.read()

                    #Making detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    #Draw for tracking
                    draw_styled_landmarks_np_nf_B(image, results)

                    #Extract landmark features
                    head =list(np.zeros(1*3))
                    lh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
                    rh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
                
                    if results.pose_landmarks:
                        for id, lm in enumerate(results.pose_landmarks.landmark):
                            h,w,c = frame.shape
                            cx, cy = int(lm.x*w), int(lm.y*h)
                
                            if id == 0:
                                if lm.visibility > 0.8:
                                    head = list(np.array([lm.x, lm.y, lm.z]))
                                else:
                                    head =list(np.zeros(1*3))
                            
                    
                    #Join the rows
                    row = lh_row + rh_row + head

                    #Sign Language Detections
                    X = pd.DataFrame([row])
                    sign_class = model_L.predict(X)[0]
                    sign_prob = model_L.predict_proba(X)[0]

                    #Sentence Logic
                    if sign_prob[np.argmax(sign_prob)] > threshold:
                        predictions.append(sign_class)

                        print(sign_class, sign_prob[np.argmax(sign_prob)])

                        if predictions[-pr:] == [sign_class]*pr:
                            if len(sentence) > 0:
                                if sign_class != sentence[-1]:
                                    sentence.append(sign_class)
                                    draw_styled_landmarks_G(image, results)
                                    if sentence[-1] in last_sign_list:
                                        sign_output(sign_list, sentence, sentence_out)
                                    if sentence[-1] in one_sign_list:
                                        sentence_out.append(sign_class)
                                    speak(sign_class)
                            else:
                                sentence.append(sign_class)
                                draw_styled_landmarks_np_nf_B(image, results)
                                if sentence[-1] in one_sign_list:
                                        sentence_out.append(sign_class)
                                speak(sign_class)


                    if len(sentence) > 3:
                            sentence = sentence[-3:]
                            

                    cv2.rectangle(image, (0,0), (640,40),(255,255,255), -1 )
                    cv2.putText(image,  ' '.join(sentence), (3,30),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                    
                    cv2.rectangle(image, (0,80), (640,40),(255,255,255), -1 )
                    cv2.putText(image,  ' '.join(sentence_out), (3,70),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                    #Show OpenCV Feed on the screen
                    cv2.imshow('OpenCV Feed', image)

                    #break loop
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
        detect(0)
     
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port='8000', debug=True)