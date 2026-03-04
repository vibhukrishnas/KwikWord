//Numpy array shape [8]
//Min -0.304912805557
//Max 0.511904060841
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
conv2d_bias_t b2[8];
#else
conv2d_bias_t b2[8] = {0.02609, 0.24075, 0.20081, -0.30491, -0.22326, 0.51190, 0.26164, 0.30265};

#endif

#endif
