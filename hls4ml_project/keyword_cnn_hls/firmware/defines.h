#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,3>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,3>, 1*1> layer17_t;
typedef ap_fixed<8,3> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer2_t;
typedef ap_fixed<8,3> conv2d_weight_t;
typedef ap_fixed<8,3> conv2d_bias_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer3_t;
typedef ap_fixed<16,6> re_lu_param_t;
typedef ap_fixed<18,8> re_lu_table_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer4_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer18_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer5_t;
typedef ap_fixed<8,3> conv2d_1_weight_t;
typedef ap_fixed<8,3> conv2d_1_bias_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer6_t;
typedef ap_fixed<16,6> re_lu_1_param_t;
typedef ap_fixed<18,8> re_lu_1_table_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer7_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer19_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer8_t;
typedef ap_fixed<8,3> conv2d_2_weight_t;
typedef ap_fixed<8,3> conv2d_2_bias_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer9_t;
typedef ap_fixed<16,6> re_lu_2_param_t;
typedef ap_fixed<18,8> re_lu_2_table_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer10_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer12_t;
typedef ap_fixed<8,3> dense_weight_t;
typedef ap_fixed<8,3> dense_bias_t;
typedef ap_uint<1> layer12_index;
typedef nnet::array<ap_fixed<8,3>, 64*1> layer13_t;
typedef ap_fixed<16,6> re_lu_3_param_t;
typedef ap_fixed<18,8> re_lu_3_table_t;
typedef nnet::array<ap_fixed<16,6>, 6*1> layer15_t;
typedef ap_fixed<8,3> dense_1_weight_t;
typedef ap_fixed<8,3> dense_1_bias_t;
typedef ap_uint<1> layer15_index;
typedef nnet::array<ap_fixed<8,3>, 6*1> result_t;
typedef ap_fixed<18,8> dense_1_softmax_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
