//
// Created by Josh Shiells on 08/08/2023.
//

#ifndef NEURALNETWORKSIM_NEURON_H
#define NEURALNETWORKSIM_NEURON_H
#include <cstdint>
#include <random>
#include "imgui.h"
namespace nnsim {

        class Neuron {
        public:
            float weight;  // Needs to be public due to ImGUI referencing
            float bias;
            Neuron(float weight, float bias) {
                this->weight = weight;
                this->bias = bias;
            }
            explicit Neuron(int random_seed) {
                std::default_random_engine generator(random_seed);
                std::uniform_real_distribution<float> distribution(-1.0, 1.0);
                this->weight = distribution(generator);
                this->bias = distribution(generator);
            }

            ~Neuron() = default;

            void draw(ImDrawList* drawList, ImVec2 center, float radius) const {
                drawList->AddCircleFilled(center, radius, IM_COL32(0, 0, 0, 255), 32);
                ImVec2 text_size = ImGui::CalcTextSize("W: %.2f", nullptr, true);
                if (text_size.y == 0) text_size.y = 1; // Avoid zero-division.
                if (text_size.y > radius * 2) text_size.y = radius * 2;
                float margin = 5.0f;
                // Find the midpoint between the 2 lines
                ImGui::SetCursorPos(ImVec2(center.x - text_size.x / 2, (center.y) - ((text_size.y))));
                ImGui::Text("W: %.2f", weight);
                ImGui::SetCursorPos(ImVec2(center.x - text_size.x / 2, (center.y+margin) - ((text_size.y) / 2)));
                ImGui::Text("B: %.2f", bias);
            };


}; // nnsim
}
#endif //NEURALNETWORKSIM_NEURON_H
