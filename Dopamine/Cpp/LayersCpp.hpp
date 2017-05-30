//
//  LayersCpp.hpp
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/11.
//  Copyright © 2017年 Runo. All rights reserved.
//

#ifndef LayersCpp_hpp
#define LayersCpp_hpp

void Layer_Sigmoid(float* res, float* left, int leftCapacity);
void Layer_SigmoidBackward(float* res, float* left, float* lastOutput, int leftCapacity);

void Layer_Tanh(float* res, float* left, int leftCapacity);
void Layer_TanhBackward(float* res, float* left, float* lastOutput, int leftCapacity);

#endif /* LayersCpp_hpp */
