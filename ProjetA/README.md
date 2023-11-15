# Gesture Recognition with TensorFlow and Arduino

Project Overview :
This project combines TensorFlow and Arduino to create a gesture recognition system using the Nano 33 Sense BLE board. The goal is to train a model to recognize "Yes (✓)" and "No (X)" gestures, integrate the model into an Arduino project, and implement a quiz where users answer questions using gestures.

Step 1: Training the Model with TensorFlow:
- Data Collection
- Data Processing
- Model Training
- Model Conversion

Step 2 : Integrating the Model with Arduino
Use the xxd converter to generate a C file from the TensorFlow Lite model.
Write Arduino code to:
- Load the TensorFlow Lite model.
- Read sensors from the Nano 33 Sense BLE for required data.
- Perform inferences using the TensorFlow Lite model.
- Detect "Yes (✓)" and "No (X)" gestures.

Step 3: Implementing the Quiz
1)Display a welcome message when the board is connected.
2)Pose three questions via the serial port and wait for the user to make the "Yes (✓)" or "No (X)" gesture.
3)Gesture Detection
(https://github.com/TayssirMrad/tinyml_tp/blob/main/ProjetA/resultat_terminal.png) 

