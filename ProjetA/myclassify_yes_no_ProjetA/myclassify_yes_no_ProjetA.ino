#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const float accelerationThreshold = 2.5; 
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name 
const char* GESTURES[] = {
  "yes",
  "no"
}; // GESTURES[1] est "no" et GESTURES[0] est "yes"

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))


float aX, aY, aZ, gX, gY, gZ;

namespace tflite {
  using TfLiteFloat32 = float; //type de données float dans l'espace de noms tflite
}

  //Funtion LookForResponse********************************************

 tflite::TfLiteFloat32* LookForResponse() {
  // Attendre un mouvement significatif
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // Lire les données d'accélération
      IMU.readAcceleration(aX, aY, aZ);

      // Somme des valeurs absolues
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // Vérifier si cela dépasse le seuil
      if (aSum >= accelerationThreshold) {
        // Réinitialiser le compteur de lectures d'échantillons
        samplesRead = 0;
        break;
      }
    }
  }
   // Vérifier si tous les échantillons requis ont été lus depuis
   // la dernière détection de mouvement significatif
 
  while (samplesRead < numSamples) {
    
    // Vérifier si de nouvelles données d'accélération ET de gyroscope sont disponibles
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) { 
      IMU.readAcceleration(aX, aY, aZ);// read the acceleration 
      IMU.readGyroscope(gX, gY, gZ);// read gyroscope data

      // Normaliser les données IMU entre 0 et 1 et les stocker 
      // dans le tenseur d'entrée du modèle
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Échec!");
          while (1);
          }
        // Return the output tensor data
        return tflOutputTensor->data.f;
      }
    }
  }
  // Return nullptr if the function does not reach the end
  return nullptr;
}
 //End of LookForResponse*********************************************


void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Bienvenue ! Répondez aux questions avec 'Yes (✓)' ou 'No (X)'.");
  
  // Obtenir la représentation TFL du tableau de modèles
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Créer un interpréteur pour exécuter le modèle
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allouer de la mémoire pour les tenseurs d'entrée et de sortie du modèle
  tflInterpreter->AllocateTensors();

  // Obtenir des pointeurs pour les tenseurs d'entrée et de sortie du modèle
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  
  int i = 0; // Init
          
  // Question 1_________________________________________________________
  Serial.println("Question 1: Est-ce que le ciel est bleu ?");
  
  // Call the LookForResponse function
  tflite::TfLiteFloat32* outputData = LookForResponse();     
     
          Serial.print("Votre response est : ");
          if (tflOutputTensor->data.f[1]<tflOutputTensor->data.f[0]) {
          Serial.print(GESTURES[0]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[0], 6);
          i++;
          }else{
          Serial.print(GESTURES[1]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[1], 6);
          }


  // Question 2_________________________________________________________
  Serial.println("Question 2: Le feu est-il chaud ?");

  // Call the LookForResponse function
  outputData = LookForResponse(); 
          
          Serial.print("Votre response est : ");
          if (tflOutputTensor->data.f[1]<tflOutputTensor->data.f[0]) {
          Serial.print(GESTURES[0]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[0], 6);
          i++;
          }
          
          else{
          Serial.print(GESTURES[1]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[1], 6);
          }
          
  // Question 3_________________________________________________________
  Serial.println("Question 3:  Les humains peuvent-ils survivre sans oxygène ?");
        
    // Call the LookForResponse function
    outputData = LookForResponse();
          
          Serial.print("Votre response est : ");
          if (tflOutputTensor->data.f[1]<tflOutputTensor->data.f[0]) {
          Serial.print(GESTURES[0]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[0], 6);
          
          }
          
          else{
          Serial.print(GESTURES[1]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[1], 6);
          i++;
          }
          Serial.print("Votre score :");// affichage du score final
          Serial.print(i);
          Serial.println();

          }
