//
//  LayersCpp.hpp
//  RunoNetTest
//
//  Created by Runo Sahara on 2017/05/11.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#ifndef LayersCpp_hpp
#define LayersCpp_hpp

// TODO: '_'を先頭につける

void Layer_Sigmoid(float* res, float* left, int leftCapacity);
void Layer_SigmoidBackward(float* res, float* left, float* lastOutput, int leftCapacity);

void Layer_Tanh(float* res, float* left, int leftCapacity);
void Layer_TanhBackward(float* res, float* left, float* lastOutput, int leftCapacity);

void _Layer_ResetZeroOrNegativeAndMakeMask(float* res, float* mask, float* left, int leftCapacity);
void _Layer_ResetZeroOrNegative(float* res, float* left, int leftCapacity);
void _Layer_ApplyMask(float* left, float* mask, int leftCapacity);


#endif /* LayersCpp_hpp */
