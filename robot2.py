import cv2
import os
import pandas as pd
import speech_recognition as sr
import pyttsx3
import streamlit as st
import threading
import queue
from pandasai import Agent

# Initialize components
recognizer = sr.Recognizer()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Adjust data length for consistent DataFrames
def adjust_length(data):
    max_length = max(len(v) for v in data.values())
    for key, value in data.items():
        if len(value) < max_length:
            data[key] = value + [None] * (max_length - len(value))
        elif len(value) > max_length:
            data[key] = value[:max_length]
    return data

# Data for two student groups
data_student_a = adjust_length({
    'sl.no': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    'register no': ['23uai002', '23uai003', '23uai004', '23uai006', '23uai008', '23uai010', '23uai016', '23uai018', 
                    '23uai019', '23uai025', '23uai026', '23uai029', '23uai030', '23uai033', '23uai036', '23uai038', 
                    '23uai041', '23uai044', '23uai045', '23uai046', '23uai047', '23uai049', '23uai050', '23uai051', 
                    '23uai052', '23uai054', '23uai055', '23uai056', '23uai057', '23uai059', '23uai061', '23uai062', 
                    '23uai068', '23uai069', '23uai070', '23uai073', '23uai074', '23uai076', '23uai082', '23uai083', 
                    '23uai086', '23uai087', '23uai089', '23uai091', '23uai092', '23uai094', '23uai097', '23uai098', 
                    '23uai103', '23uai108', '23uai109', '23uai113', '23uai114', '23uai116', '23uai117', '23uai118', 
                    '23uai119', '23uai120', '23ail002', '23ail003', '23ail004'],
    'enroll no': [231332, 230712, 230634, 231321, 231562, 230864, 230829, 231217, 
                  231366, 230685, 230584, 231555, 230337, 231102, 230830, 230085, 
                  230367, 231168, 231055, 230383, 230735, 230300, 230369, 231351, 
                  231135, 230890, 230172, 231305, 231358, 230677, 230192, 230264, 
                  231091, 231036, 230841, 230483, 230896, 231016, 231225, 230996, 
                  231174, 230116, 230202, 231245, 230986, 230202, 230566, 230344, 
                  231413, 230978, 231606, 230217, 230206, 230409, 231004, 231224, 
                  231239, 230592, 240019, 240009, 241015],
    'name': ['abbhishek behera', 'abdulkalam j', 'abieswar t', 'aisva malar a', 'akhilesh b', 
             'al raafhath r k', 'arun srinivas a', 'aswitha alias swetha k', 'bharanidharan t', 
             'devisri i', 'dhaksha charan r', 'divyanand m', 'ganesh deepak n', 'guralarasan g', 
             'harini a', 'harini p (16.04.2005)', 'hemajothi s', 'jaidhar s', 'janani n', 'jassem', 
             'kanimozhi s', 'karthik m', 'karthikeyan v', 'kavitha s', 'kaviya p s', 'logapriya b', 
             'logeshwar r', 'lokitha k', 'mahesh r', 'manojkumar p', 'mithun k', 'mohamed nafeez s', 
             'narmadha d', 'navaneethakrishnan j', 'naveen kumar s', 'nitish m', 'nityasri r s', 
             'prabhanjan j', 'praveen kumaran p', 'priya darshini k', 'ragul t', 'rathnaprasad d', 
             'rayyan', 'rindhiya a', 'rohan kumar k v', 'sanjivkumar j', 'shivaani vaithiyanathan ganesh', 
             'siddarth s', 'subitsha s', 'thanush k', 'tharaniya s', 'vidhesh d', 'vignesh vijairaj s', 
             'vishal s', 'vishnu prasath p g', 'vishnulakshmi a', 'vishwa p', 'yadu raj a']
})  # Fill with your data
data_student_b = adjust_length({   'sl.no': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    'enroll number': [230365, 230851, 230320, 230984, 231318, 231699, 230040, 230466, 231361, 230782, 231685, 230885, 231081, 231445, 231347, 230273, 231303, 231574, 230886, 231044, 230973, 230607, 230543, 230262, 231526, 230825, 230853, 230961, 231762, 230880, 230889, 230970, 230154, 230362, 231406, 231147, 231027, 230696, 231340, 230470, 231108, 231561, 230504, 230291, 231603, 231195, 231392, 231278, 230989, 230377, 231596, 231052, 230760, 231315, 231177, 230447, 230903, 230877, 230424, 231593],
    'register number': ['23uai001', '23uai005', '23uai007', '23uai009', '23uai011', '23uai012', '23uai013', '23uai014', '23uai017', '23uai015', '23uai020', '23uai021', '23uai022', '23uai023', '23uai024', '23uai027', '23uai028', '23uai031', '23uai034', '23uai035', '23uai037', '23uai039', '23uai040', '23uai042', '23uai043', '23uai048', '23uai053', '23uai058', '23uai060', '23uai063', '23uai064', '23uai065', '23uai066', '23uai067', '23uai071', '23uai072', '23uai075', '23uai077', '23uai079', '23uai080', '23uai081', '23uai084', '23uai085', '23uai088', '23uai090', '23uai093', '23uai095', '23uai096', '23uai099', '23uai100', '23uai101', '23uai102', '23uai104', '23uai105', '23uai106', '23uai107', '23uai110', '23uai111', '23uai112', '23uai115'],
    'name': ['aadhithya', 'agalya', 'ajay govind', 'akshya', 'amirthabhanu', 'andrew surjit ronald f', 'annie maerlin', 'anushree', 'aravind raj', 'arjun', 'bhavesh', 'bhuvanesh', 'chandra arul nishanthini', 'daamini', 'dakshnamorthy', 'dhavanesh', 'dhivya', 'ganesh', 'hafzafarzana', 'harija', 'harini', 'harish', 'hemachandran', 'hemeshwaran', 'indhuja', 'karmugilan', 'lalithambigha', 'manikandan', 'meenaloshani', 'mohammed aashiq', 'mohana priya', 'mordheeshvara', 'mugaesh', 'muthu krishna', 'nithya shri', 'nithya sri', 'prabhakaran', 'pradeepraj', 'pragadeeswaran', 'pramila', 'pratheeb', 'priyamadhan', 'priyanka', 'ravikanth', 'reyash', 'sangaradas', 'sathiyam', 'sarathy', 'sivaranjani', 'sreevardhini', 'sri aishwarya', 'srivathsan', 'sudharsan', 'sushma saraswathi', 'sushmidha', 'tarunraj', 'theenash', 'uthradevi', 'venkatesh', 'vishaal'],
})  # Fill with your data

df_a = pd.DataFrame(data_student_a)
df_b = pd.DataFrame(data_student_b)
PANDASAI_API_KEY = st.secrets["PANDASAI_API_KEY"]
agent = Agent([df_a, df_b])

# TTS setup
engine = pyttsx3.init()
tts_queue = queue.Queue()
tts_lock = threading.Lock()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)

def stop_tts():
    tts_queue.put(None)
    tts_thread.join()

# Voice recognition
def listen_for_command():
    with sr.Microphone() as source:
        st.write("Listening for voice command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            st.write(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand the command.")
            return None
        except sr.RequestError:
            st.write("Could not request results from Google Speech Recognition service.")
            return None

# Streamlit app setup
st.title("Optimized Voice-Controlled Face Detection and Data Query System")

if st.button('Start Face Detection'):
    st.write("Face detection started!")
    cap = cv2.VideoCapture(1)  # Use the first available camera

    # Optimize camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.write("Error: Could not access the webcam.")
    else:
        first_face = None
        first_face_saved = False
        tolerance = 50  # Adjust for broader face tracking
        voice_assistant_triggered = False
        frame_count = 0  # Frame skipping mechanism

        st.write("Press 'q' in the webcam window to exit the program.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Failed to read frame from webcam.")
                break

            # Skip frames to reduce processing load
            if frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    if not first_face_saved:
                        first_face = (x, y, w, h)
                        first_face_saved = True
                        st.write("First face detected and saved at:", first_face)
                        speak("Face detected. What would you like me to do next?")
                        voice_assistant_triggered = True

                    if first_face and abs(x - first_face[0]) <= tolerance and abs(y - first_face[1]) <= tolerance:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Tracking First Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "Ignoring Other Faces", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            frame_count += 1

            if voice_assistant_triggered:
                command = listen_for_command()
                if command:
                    st.write(f"Voice command received: {command}")
                    response = agent.chat(command)
                    st.write(f"AI Response: {response}")
                    speak(f"Here is the answer: {response}")
                    if "exit" in command:
                        st.write("Exiting program as per your request.")
                        break

            cv2.imshow('Optimized Face Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                st.write("Exiting program.")
                break

        cap.release()
        cv2.destroyAllWindows()

stop_tts()
