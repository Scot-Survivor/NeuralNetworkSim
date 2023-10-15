//
// Created by Josh Shiells on 08/08/2023.
//

#ifndef NEURALNETWORKSIM_DENSELAYER_H
#define NEURALNETWORKSIM_DENSELAYER_H
#include <cstdint>
#include <vector>
#include <CL/sycl.hpp>
#include "Neuron.h"
#include "Structs.h"

namespace nnsim {

    class DenseLayer {
    private:
        uint64_t n_inputs;
        uint64_t n_neurons;
        std::vector<Neuron> neurons;
        int random_seed = 123;
        std::vector<ImVec2> pos_of_drawn_neurons;
        std::vector<NeuronLinkDrawData> lines;
        std::vector<NeuronDrawData> neuron_draw_data;
    public:
        DenseLayer(uint64_t n_inputs, uint64_t n_neurons);
        DenseLayer(uint64_t n_inputs, uint64_t n_neurons, int random_seed);
        ~DenseLayer() = default;

        void forward(sycl::queue q, sycl::buffer<float> input_data, sycl::buffer<float> output_data);
        void draw(ImVec2 center, float radius_of_neurons, float padding);
        void draw(ImVec2 center, float radius_of_neurons, float padding, const std::vector<ImVec2>& pos_of_input);
        void draw_lines(ImDrawList* drawList);
        void draw_neurons(ImDrawList* drawList);

        int get_n_neurons() const { return n_neurons; }

        std::vector<ImVec2> get_pos_of_drawn_neurons() const { return pos_of_drawn_neurons; }
    };

} // nnsim

#endif //NEURALNETWORKSIM_DENSELAYER_H
