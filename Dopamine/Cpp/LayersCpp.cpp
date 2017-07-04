//
//  SimpleLayersCpp.cpp
//  RunoNetTest
//
//  Created by Runo Sahara on 2017/05/11.
//  Copyright © 2017 Runo Sahara. All rights reserved.
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
	

void _Layer_ResetZeroOrNegativeAndMakeMask(float* res, float* mask, float* left, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		float val = *left;
		int gtZero = val > 0.0f;
		if (gtZero) {
			*res = val;
		} else {
			*(int*)res = 0;
		}
		*(int*)mask = gtZero;
		res++;
		mask++;
	}
	
}

void _Layer_ResetZeroOrNegative(float* res, float* left, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		float val = *left;
		if (*left > 0.0f) {
			*res = val;
		} else {
			*(int*)res = 0;
		}
		res++;
	}
	
}

void _Layer_ApplyMask(float* left, float* mask, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		if (*(int*)mask == 0) {
			*(int*)left = 0;
		}
		mask++;
	}
	
}


}
