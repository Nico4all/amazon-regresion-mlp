#include <iostream>
#include <cmath>

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

int main() {

    std::cout << "=== Emulacion MLP Amazon Sale Report ===\n";
    std::cout << "Arquitectura: 47 -> 32 -> 16 -> 1\n\n";

    for (int s = 0; s < TEST_SAMPLES; s++) {

        float pred = predictAmount(TEST_INPUTS[s]);
        float expected_real = TEST_EXPECTED[s];
        float tf_ref = TF_REFERENCE_OUTPUTS[s];

        float diff_real = fabs(pred - expected_real);
        float diff_tf = fabs(pred - tf_ref);

        std::cout << "Caso " << s + 1 << "\n";

        std::cout << "Valor real dataset: " << expected_real << "\n";
        std::cout << "Referencia TensorFlow: " << tf_ref << "\n";
        std::cout << "Prediccion Arduino: " << pred << "\n";

        std::cout << "Error vs valor real: " << diff_real << "\n";
        std::cout << "Error vs TensorFlow: " << diff_tf << "\n";

        std::cout << "----------------------------\n";
    }

    return 0;
}