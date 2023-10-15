//
// Created by Josh Shiells on 08/08/2023.
//

#ifndef NEURALNETWORKSIM_STRUCTS_H
#define NEURALNETWORKSIM_STRUCTS_H
#include "imgui.h"
namespace nnsim {
    struct NeuronLinkDrawData {
        ImVec2 p1;
        ImVec2 p2;
        ImU32 color{};
        float thickness{};
    };

    struct NeuronDrawData {
        ImVec2 vec;
        float radius{};
    };
}
#endif //NEURALNETWORKSIM_STRUCTS_H
