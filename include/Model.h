//
// Created by Josh Shiells on 08/08/2023.
//

#ifndef NEURALNETWORKSIM_MODEL_H
#define NEURALNETWORKSIM_MODEL_H
#include "DenseLayer.h"
#include "vector"
namespace nnsim {

    class Model {
    private:
        std::vector<DenseLayer> layers;
        float layer_padding = 32;
        float neuron_radius = 32;
        float neuron_padding = 10;
    public:
        explicit Model(std::vector<DenseLayer> layers);
        Model();
        ~Model() = default;

        void add(const DenseLayer& layer);
        void forward(const sycl::queue& q, std::vector<float> &inputs, std::vector<float> &outputs);
        void draw(ImDrawList* drawList);
        void set_neuron_padding(float padding) { layer_padding = padding; }
        int get_largest_layer();
    };

} // nnsim

#endif //NEURALNETWORKSIM_MODEL_H
