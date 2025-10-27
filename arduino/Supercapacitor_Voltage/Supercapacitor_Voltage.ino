/*
  ================================================================================
  -------------- Arduino Nicla Voice: Supercapacitor Voltage Reader --------------
  ================================================================================
  
  This firmware is for the "Worker" device (Arduino Nicla Voice) in our
  energy-harvesting project.
  
  Its only job is to:
  1. Read the voltage of the supercapacitor via an Analog-to-Digital (ADC) pin.
  2. Transmit this voltage value over USB Serial to the host PC.
  
  This data will be read by our Python "Manager" script.
  
  =================================================================================
  !!! -------------- CRITICAL HARDWARE WARNING - READ THIS FIRST -------------- !!!
  =================================================================================
  
  The Arduino Nicla Voice's ADC pins (like A0) can only safely read a maximum of 1.8V.
  
  The supercapacitor will be charged to 4.3V or more, which will
  **INSTANTLY DESTROY** the Arduino board if connected directly to an analog pin.
  
  We **MUST** use a VOLTAGE DIVIDER to read the voltage safely.
  
  A 1:3 voltage divider with a 20k-ohm (R_TOP) and 10k-ohm (R_BOTTOM) resistor will
  divide the voltage by 3, bringing it into a safe range (4.3V becomes ~1.43V).
  
  Connections:
  
  (Supercap +) ---- [ R_TOP (20k) ] ---- (A0 Pin) ---- [ R_BOTTOM (10k) ] ---- (GND)
  
  
  This code assumes having this 1:3 voltage divider in place.
  It multiplies the measured voltage by 3 (the VOLTAGE_DIVIDER_FACTOR) to get the real value.

  ================================================================================
*/

  //*******************//
 // --- Constants --- //
//*******************//

// Define the ADC pin connected to the voltage divider's midpoint.
const int VOLTAGE_PIN = A0;

// The ADC resolution on the Nicla Voice is 10-bit (0-1023).
const float MAX_ADC_VALUE = 1023.0;

// The default analog reference voltage is 3.3V.
const float ADC_REFERENCE_VOLTAGE = 3.3;

// --- Voltage Divider Resistors ---
// Define the resistor values used in the voltage divider circuit.
const float R_TOP = 20000.0; // The "top" resistor from (Supercap +) to (A0 Pin)
const float R_BOTTOM = 10000.0; // The "bottom" resistor from (A0 Pin) to (GND)

// The multiplication factor from our voltage divider.
/*
  Standard formula: V_out = V_in * (R_BOTTOM / (R_TOP + R_BOTTOM))
  We need to find V_in (the supercap voltage):
    V_in = V_out * (R_TOP + R_BOTTOM) / R_BOTTOM
*/
// This is our multiplication factor:
const float VOLTAGE_DIVIDER_FACTOR = (R_TOP + R_BOTTOM) / R_BOTTOM;

// How often to send data (in milliseconds).
const int SEND_INTERVAL_MS = 1000;

  //************************//
 // --- Setup Function --- //
//************************//

// This runs once when the board powers on.
void setup() {
  // Initialize the USB Serial port at 9600 baud.
  Serial.begin(9600);
  
  // Wait for Serial Monitor to open.
  while (!Serial); 
  
  // Set the ADC resolution to 10 bits (0-1023).
  analogReadResolution(10);
  
  Serial.println("Nicla Voice: Voltage Reader Initialized.");
}

  //***********************//
 // --- Loop Function --- //
//***********************//

void loop() {
  // 1. Read the raw 10-bit value from the ADC pin (0-1023).
  int rawADCValue = analogRead(VOLTAGE_PIN);
  
  // 2. Convert the raw value into the voltage AT THE PIN (0V - 3.3V).
  float pinVoltage = (rawADCValue / MAX_ADC_VALUE) * ADC_REFERENCE_VOLTAGE;
  
  // 3. Convert the pin voltage to the ACTUAL supercapacitor voltage by applying our voltage divider's math.
  float supercapVoltage = pinVoltage * VOLTAGE_DIVIDER_FACTOR;
  
  // 4. Send the final, calculated voltage to the host PC.
  //    We use println() so the Python script can read it line-by-line.
  Serial.println(supercapVoltage);
  
  // 5. Wait for the defined interval before sending again.
  delay(SEND_INTERVAL_MS);
}
