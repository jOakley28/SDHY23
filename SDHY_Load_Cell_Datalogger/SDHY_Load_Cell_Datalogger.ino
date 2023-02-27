//--------------------------------------- SD Card Library Setup ---------------------------------------
#include <SPI.h>
#include <SD.h>
#include <HX711.h>

//---------------------------- Setup for EEPROM to save calibration for sensor ------------------------
#include <EEPROM.h>
int eeAddress = 0;
#define EEPROM_SIZE 100

struct Calibration {  // create structure to save into EEPROM
  float scale;        // scale variable
  long offset;        // offset variable (tare)
};


Calibration calibration;  // create an instance of the Calibration structure. Name that instance "calibration"

// Custom function SaveStruct(int, Calibration)
// This function uses the EEPROM library to save values to EEPROM
// Arguments are the EEPROM address where we would like to store the data
// and an instance of the Calibration struct
void SaveStruct(int eeAddress, Calibration calibration) {
  EEPROM.put(eeAddress, calibration);
  // Serial.println("Save Calibration object to EEPROM: ");
  // Serial.print("Scale: ");
  // Serial.println(calibration.scale);
  // Serial.print("Offset: ");
  // Serial.println(calibration.offset);
}

// Custom function LoadStruct()
// This function uses the EEPROM library to pull previously saved values from EEPROM
// Arguments are the EEPROM address where we would like to pulll the data from
// This function uses EEPROM.get() to save the data from the EEPROM address sepcified to the variable "calibration"
Calibration LoadStruct(int eeAddress) {
  EEPROM.get(eeAddress, calibration);
  // Serial.println("Read custom object from EEPROM: ");
  // Serial.print("Scale: ");
  // Serial.println(calibration.scale);
  // Serial.print("Sffset: ");
  // Serial.println(calibration.offset);
  return calibration;
}
// -------------------------------- End of EEPROM stuff --------------------------------

const int chipSelect = 10;  // this value must be 8 for the Sparkfun SD Shield
String fileName = "log_0.txt";

String dataString = "";

const int greenLED = 6;
const int buttonPin = 7;
const int tareButton = 9;
const int selectorSwitch = 8;

boolean greenState = 1;

int state = 0;
int runCount = 0;
boolean buttonState = true;

unsigned long startTime = 0;
unsigned long runTime = 0;

volatile float reading;

HX711 sensor;
uint8_t VCC = 2;
uint8_t clockPin = 3;
uint8_t dataPin = 4;
uint8_t GND = 5;

float calibrationWeight = 100;  // weight to use for calibration

//================================================ Setup Function ==================================================
void setup() {

  float scale;
  long offset;

  Serial.begin(9600);  // Serial on
  Serial.println("Serial Open");

  pinMode(GND, OUTPUT);  // power on the HX711
  pinMode(VCC, OUTPUT);
  digitalWrite(GND, LOW);
  digitalWrite(VCC, HIGH);
  Serial.println("HX711 On");

  // button setup
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(tareButton, INPUT_PULLUP);
  pinMode(selectorSwitch, INPUT_PULLUP);
  Serial.println("Buttons intialized");

  pinMode(greenLED, OUTPUT);
  digitalWrite(greenLED, HIGH);
  delay(4000);
  digitalWrite(greenLED, LOW);
  delay(250);
  digitalWrite(greenLED, HIGH);
  delay(250);


  sensor.begin(dataPin, clockPin);


  // Select which sensor to read EEPROM value for
  if (digitalRead(selectorSwitch) == LOW) {
    calibration = LoadStruct(0);  // load EEPROM for larger load cell
    Serial.println("BIG load cell selected");
  } else {
    calibration = LoadStruct(500);  // load EEPROM for button cell
    Serial.println("SMALL load cell selected");
  }


  if (digitalRead(tareButton) == LOW) {
    startTime = millis();
    Serial.println("Entered Calibration Mode");
    Serial.println("Preparing to tare the sensor");
    while (startTime - millis() < 5000) {
      digitalWrite(greenLED, HIGH);
      delay(50);
      digitalWrite(greenLED, LOW);
      delay(50);
    }

    Serial.println("Taring");

    sensor.tare();

    Serial.println("Sensor tared");

    Serial.print("Place ");
    Serial.print(calibrationWeight);
    Serial.println(" kg on the sensor");

    while (digitalRead(tareButton) == HIGH) {
      analogWrite(greenLED, 255 * sin(millis() / 1000.0));
      Serial.println("Waiting for button press");
    }


    Serial.println("Calibrating");

    sensor.calibrate_scale(calibrationWeight, 5);

    Serial.println("Calibration complete");

    scale = sensor.get_scale();
    offset = sensor.get_offset();

    calibration.offset = offset;
    calibration.scale = scale;

    Serial.print("Scale: ");
    Serial.println(scale);
    Serial.print("Offset: ");
    Serial.println(offset);

    if (digitalRead(selectorSwitch) == LOW) {
      SaveStruct(0, calibration);
      Serial.println("EEPROM saved for big load cell");
    } else {
      SaveStruct(500, calibration);
      Serial.println("EEPROM saved for small load cell");
    }

    Serial.println("EEPROM saved");
  }



  sensor.set_scale(calibration.scale);    // read scale from eeprom position 0 or 500
  sensor.set_offset(calibration.offset);  // read offset from eeprom position 100 or 600

  Serial.print("Scale: ");
  Serial.println(calibration.scale);
  Serial.print("Offset: ");
  Serial.println(calibration.offset);


  //--------------------------------------------- SD Setup and Init. Code -----------------------------------------//
  if (!SD.begin(chipSelect)) {                      // initialize SD card
    Serial.println("Card failed, or not present");  // if it doesn't work let us know and stop the code
    // don't do anything more:
    digitalWrite(greenLED, LOW);
    while (1) {
      analogWrite(greenLED, sin(millis() / 100));
    }
  }
  Serial.println("card initialized.");
  state = 0;
}



//==================================================== Loop Function =======================================================
void loop() {
  switch (state) {
    case 0:  // Waiting state; alternate green and red LEDs until user pressed button, then move to state 1;
      if (millis() % 500 == 0) {
        greenState = !greenState;
        digitalWrite(greenLED, greenState);
        // Serial.println("Waiting");
      }
      if (digitalRead(buttonPin) == LOW) {
        delay(250);
        state = 1;
      }
      break;

    case 1:
      Serial.println("Moving to Logging State");
      while (SD.exists(fileName)) {
        Serial.println("File Already Exists!");
        runCount++;
        fileName = "log_";
        fileName += String(runCount);
        fileName += ".txt";
      }

      state = 2;
      startTime = millis();
      newFile(fileName);

      Serial.print("New file name = ");
      Serial.println(fileName);

      break;

    case 2:

      logData();
      if (digitalRead(buttonPin) == LOW) {
        delay(250);
        state = 0;
        Serial.println("Datalogging session over!");
      }
      break;
  }
}


void newFile(String fileName) {
  File dataFile = SD.open(fileName, FILE_WRITE);  // open the datafile
  delay(100);

  if (dataFile) {                           // if the file is available
    dataFile.println("Time\tADC_Reading");  // write datafile header to it
    dataFile.close();                       // close datafile


    Serial.print("Filename: ");
    Serial.println(fileName);
    Serial.println("Time\tADC_Reading");  // print to the serial port too

    digitalWrite(greenLED, HIGH);

  } else {  // if the file isn't open, pop up an error
    Serial.print("error opening ");
    Serial.println(fileName);
    digitalWrite(greenLED, LOW);
    while (1) {
      analogWrite(greenLED, sin(millis() / 100));
    }
  }
}



// =================================================== DATALOGGING SECTION ====================================================
// =============================================================================================================================
void logData() {
  //digitalWrite(greenLED, HIGH);
  reading = sensor.get_units();

  digitalWrite(greenLED, LOW);  // turn off green LED for blink

  runTime = millis() - startTime;  // set current time for this datapoint

  //------------------------------------------------- Data Logging -----------------------------------------------------
  dataString = "";                         // make a string for assembling the data to log
  dataString += String(runTime / 1000.0);  // time
  dataString += "\t";                      // (tab)
  dataString += String(reading);           // sensor reading (kg)

  //------------------------------------------------- Write to SD Card ------------------------------------------------

  File dataFile = SD.open(fileName, FILE_WRITE);  // open the datafile
  if (dataFile) {                                 // if the file is available
    dataFile.println(dataString);                 // write to it
    dataFile.close();                             // close datafile

    Serial.println(dataString);  // print to the serial port too

    digitalWrite(greenLED, HIGH);
  } else {  // if the file isn't open
    Serial.print("error opening ");
    Serial.println(fileName);  // throw an error

    digitalWrite(greenLED, LOW);
  }
}