#include "model_data.h"
#include "scaler_data.h"
#include "test_cases.h"
#include "tf_reference_cases.h"

float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

float predictScaled(const float input[INPUT_SIZE]) {
  float h1[HIDDEN1_SIZE];
  float h2[HIDDEN2_SIZE];
  float out[OUTPUT_SIZE];

  for (int j = 0; j < HIDDEN1_SIZE; j++) {
    float sum = b1[j];
    for (int i = 0; i < INPUT_SIZE; i++) {
      sum += input[i] * W1[i][j];
    }
    h1[j] = relu(sum);
  }

  for (int j = 0; j < HIDDEN2_SIZE; j++) {
    float sum = b2[j];
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      sum += h1[i] * W2[i][j];
    }
    h2[j] = relu(sum);
  }

  for (int j = 0; j < OUTPUT_SIZE; j++) {
    float sum = b3[j];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      sum += h2[i] * W3[i][j];
    }
    out[j] = sum;
  }

  return out[0];
}

float inverseScale(float y_scaled) {
  return y_scaled * TARGET_SCALE + TARGET_MEAN;
}

float predictAmount(const float input[INPUT_SIZE]) {
  return inverseScale(predictScaled(input));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }

  Serial.println("=== Emulacion MLP Amazon Sale Report ===");
  Serial.println("Arquitectura: 47 -> 32 -> 16 -> 1");
  Serial.println();

  for (int s = 0; s < TEST_SAMPLES; s++) {
    float pred = predictAmount(TEST_INPUTS[s]);
    float expected_real = TEST_EXPECTED[s];
    float tf_ref = TF_REFERENCE_OUTPUTS[s];

    float diff_real = pred - expected_real;
    if (diff_real < 0) diff_real = -diff_real;

    float diff_tf = pred - tf_ref;
    if (diff_tf < 0) diff_tf = -diff_tf;

    Serial.print("Caso ");
    Serial.println(s + 1);

    Serial.print("Valor real dataset: ");
    Serial.println(expected_real, 6);

    Serial.print("Referencia TensorFlow: ");
    Serial.println(tf_ref, 6);

    Serial.print("Prediccion Arduino: ");
    Serial.println(pred, 6);

    Serial.print("Error vs valor real: ");
    Serial.println(diff_real, 6);

    Serial.print("Error vs TensorFlow: ");
    Serial.println(diff_tf, 6);

    Serial.println("----------------------------");
  }
}

void loop() {
}