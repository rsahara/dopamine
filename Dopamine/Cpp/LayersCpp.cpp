//
//  SimpleLayersCpp.cpp
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/11.
//  Copyright © 2017年 Runo. All rights reserved.
//

#include <cmath>

extern "C" {

#include "LayersCpp.hpp"

void Layer_Sigmoid(float* res, float* left, int leftCapacity) {

	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		*res = 1.0f / (1.0f + expf(- *left));
		res++;
	}

}

void Layer_SigmoidBackward(float* res, float* left, float* lastOutput, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		
		float out = *lastOutput;
		*res = *left * out * (1.0f - out);

		res++;
		lastOutput++;
	}

}

void Layer_Tanh(float* res, float* left, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		*res = tanhf(*left);
		res++;
	}
	
}

void Layer_TanhBackward(float* res, float* left, float* lastOutput, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		
		float out = *lastOutput;
		*res = *left * (1.0f - out * out);
		
		res++;
		lastOutput++;
	}
	
}

}
