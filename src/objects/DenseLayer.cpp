//
// Created by Josh Shiells on 08/08/2023.
//

#include "DenseLayer.h"

namespace nnsim {
    DenseLayer::DenseLayer(uint64_t n_inputs, uint64_t n_neurons) {
        this->n_inputs = n_inputs;
        this->n_neurons = n_neurons;
        for (uint64_t i = 0; i < n_neurons; i++) {
            neurons.emplace_back(Neuron(this->random_seed));
        }

    }

    DenseLayer::DenseLayer(uint64_t n_inputs, uint64_t n_neurons, int random_seed) {
        this->n_inputs = n_inputs;
        this->n_neurons = n_neurons;
        for (uint64_t i = 0; i < n_neurons; i++) {
            neurons.emplace_back(Neuron(this->random_seed));
        }
        this->random_seed = random_seed;
    }

    void DenseLayer::draw(ImVec2 center, float radius_of_neurons, float padding) {
        float x = center.x - (n_neurons * (radius_of_neurons * 2 + padding)) / 2;
        pos_of_drawn_neurons.clear();
        neuron_draw_data.clear();
        for (uint64_t i = 0; i < n_neurons; i++) {
            auto vec = ImVec2(x + (radius_of_neurons * 2 + padding) * i, center.y);
            pos_of_drawn_neurons.push_back(vec);
            neuron_draw_data.emplace_back(nnsim::NeuronDrawData{vec, radius_of_neurons});
        }
    }

    void DenseLayer::draw(ImVec2 center, float radius_of_neurons, float padding, const std::vector<ImVec2>& pos_of_input)  {
        // Draw neurons and lines from each input to each neuron
        float x = center.x - (n_neurons * (radius_of_neurons * 2 + padding)) / 2;

        lines.clear();
        for (ImVec2 pos : pos_of_input) {
            for (uint64_t i = 0; i < n_neurons; i++) {
                lines.emplace_back(nnsim::NeuronLinkDrawData{pos, ImVec2(x + (radius_of_neurons * 2 + padding) * i, center.y), IM_COL32(255, 255, 255, 255), 1});
            }
        }
        draw(center, radius_of_neurons, padding);

    }

    void DenseLayer::draw_lines(ImDrawList *drawList) {
        for (NeuronLinkDrawData data : lines) {
            drawList->AddLine(data.p1, data.p2, data.color, data.thickness);
        }
    }

    void DenseLayer::draw_neurons(ImDrawList* drawList) {
        for (NeuronDrawData data : neuron_draw_data) {
            neurons[0].draw(drawList, data.vec, data.radius);
        }

    }

    void DenseLayer::forward(sycl::queue q, sycl::buffer<float> input_data, sycl::buffer<float> output_data) {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor input_accessor(input_data, cgh, sycl::read_only);
            sycl::accessor output_accessor(output_data, cgh, sycl::write_only);
            cgh.parallel_for(sycl::range<1>(n_neurons), [=](sycl::id<1> idx) {
                float sum = 0;
                for (int i = 0; i < input_accessor.size(); i++) {
                    sum += i; // input_accessor[i] * this->neurons[idx[0]].weight;
                }
                sum += 10; // this->neurons[idx[0]].bias;
                output_accessor[idx[0]] = sum;
            });
        });
    }
} // nnsim