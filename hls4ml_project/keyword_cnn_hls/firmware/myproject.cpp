#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &input_layer,
    hls::stream<result_t> &layer16_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_layer,layer16_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv2d_weight_t, 72>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv2d_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2d_1_weight_t, 1152>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv2d_1_bias_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<conv2d_2_weight_t, 4608>(w8, "w8.txt");
        nnet::load_weights_from_txt<conv2d_2_bias_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<dense_weight_t, 153600>(w12, "w12.txt");
        nnet::load_weights_from_txt<dense_bias_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 384>(w15, "w15.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 6>(b15, "b15.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=5292

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=4960

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=4960

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1240

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=1408

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1240

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1240

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=310

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=396

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=310

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=310

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=75

    auto& layer11_out = layer10_out;
    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1

    nnet::zeropad2d_cl<input_t, layer17_t, config17>(input_layer, layer17_out); // zp2d_conv2d

    nnet::conv_2d_cl<layer17_t, layer2_t, config2>(layer17_out, layer2_out, w2, b2); // conv2d

    nnet::thresholded_relu<layer2_t, re_lu_param_t, layer3_t, thresholdedrelu_config3>(layer2_out, 0.0, layer3_out); // re_lu

    nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out); // max_pooling2d

    nnet::zeropad2d_cl<layer4_t, layer18_t, config18>(layer4_out, layer18_out); // zp2d_conv2d_1

    nnet::conv_2d_cl<layer18_t, layer5_t, config5>(layer18_out, layer5_out, w5, b5); // conv2d_1

    nnet::thresholded_relu<layer5_t, re_lu_1_param_t, layer6_t, thresholdedrelu_config6>(layer5_out, 0.0, layer6_out); // re_lu_1

    nnet::pooling2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // max_pooling2d_1

    nnet::zeropad2d_cl<layer7_t, layer19_t, config19>(layer7_out, layer19_out); // zp2d_conv2d_2

    nnet::conv_2d_cl<layer19_t, layer8_t, config8>(layer19_out, layer8_out, w8, b8); // conv2d_2

    nnet::thresholded_relu<layer8_t, re_lu_2_param_t, layer9_t, thresholdedrelu_config9>(layer8_out, 0.0, layer9_out); // re_lu_2

    nnet::pooling2d_cl<layer9_t, layer10_t, config10>(layer9_out, layer10_out); // max_pooling2d_2

    nnet::dense<layer10_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense

    nnet::thresholded_relu<layer12_t, re_lu_3_param_t, layer13_t, thresholdedrelu_config13>(layer12_out, 0.0, layer13_out); // re_lu_3

    nnet::dense<layer13_t, layer15_t, config15>(layer13_out, layer15_out, w15, b15); // dense_1

    nnet::softmax<layer15_t, result_t, softmax_config16>(layer15_out, layer16_out); // dense_1_softmax

}

