//
// Created by Josh Shiells on 08/08/2023.
//

#include "Model.h"

#include <utility>

namespace nnsim {
    Model::Model(std::vector<DenseLayer> layers) {
        this->layers = std::move(layers);
    }

    Model::Model() {
        this->layers = std::vector<DenseLayer>();
    }

    void Model::add(const DenseLayer& layer) {
        this->layers.push_back(layer);
    }

    void Model::draw(ImDrawList *drawList) {
        float neuron_padding = layer_padding + neuron_radius;
        int num_neurons = get_largest_layer();
        for (int i = 0; i < layers.size(); i++) {
            auto pos = ImVec2(ImGui::GetIO().DisplaySize.x / (float) num_neurons,  (ImGui::GetIO().DisplaySize.y / (float) num_neurons) + (neuron_padding * (float) i));
            if (i == 0) {
                layers[i].draw(pos, neuron_radius, this->neuron_padding);
            } else {
                layers[i].draw(pos, neuron_radius, this->neuron_padding, layers[i - 1].get_pos_of_drawn_neurons());
            }
            layers[i].draw_lines(drawList);
        }
        for (auto & layer : layers) {
            layer.draw_neurons(drawList);
        }
    }

    int Model::get_largest_layer() {
        DenseLayer largest_layer = layers[0];
        for (DenseLayer layer : layers) {
            if (layer.get_n_neurons() > largest_layer.get_n_neurons()) {
                largest_layer = layer;
            }
        }
        return largest_layer.get_n_neurons();
    }

    void Model::forward(const sycl::queue& q, std::vector<float> &inputs, std::vector<float> &outputs) {
        sycl::buffer<float> input_buffer(inputs.data(), inputs.size());
        sycl::buffer<float> output_buffer(outputs.data(), outputs.size());

        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                layers[i].forward(q, input_buffer, output_buffer);
            } else {
                layers[i].forward(q, output_buffer, output_buffer);
            }
        }
    }

} // nnsim