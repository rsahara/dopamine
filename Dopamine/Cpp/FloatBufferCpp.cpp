//
//  FloatBufferCpp.cpp
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/08.
//  Copyright © 2017年 Runo. All rights reserved.
//

#include <cmath>
#include <cstring>
#include <random>

#include <Accelerate/Accelerate.h>

extern "C" {

#include "FloatBufferCpp.hpp"

void _FloatBuffer_FillZero(float* res, int length) {
	memset(res, 0, length * sizeof (float));
}

void _FloatBuffer_FillRandomGaussian(float* res, int length) {
	std::random_device device;
	std::mt19937 generator(device());
	std::normal_distribution<float> dist;

	float* resEnd = res + length;
	for (; res < resEnd; res++) {
		*res = dist(generator);
	}
}

void FloatBuffer_MatMul(float* res, float* left, float* right, int leftHeight, int leftWidth, int rightWidth) {

#if 1
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, leftHeight, rightWidth, leftWidth, 1.0f, left, leftWidth, right, rightWidth, 0.0, res, rightWidth);
	
#else
	
	float* leftEnd = left + (leftHeight * leftWidth);
	float* rightHeadEnd = right + rightWidth;
	while (left < leftEnd) {
		float* leftHeadEnd = left + leftWidth;

		for (float* rightHeadStart = right; rightHeadStart < rightHeadEnd; rightHeadStart++) {
			
			float temp = 0.0f;

			float* leftHead = left;
			float* rightHead = rightHeadStart;
			
			while (leftHead < leftHeadEnd) {
				temp += *leftHead * *rightHead;
				leftHead++;
				rightHead += rightWidth;
			}
			
			*res = temp;
			res++;
		}
		
		left = leftHeadEnd;
	}
	
#endif

}

void FloatBuffer_Mul(float* left, float* right, int leftCapacity, int rightCapacity) {
	
	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left *= *rightHead;
			left++;
		}
	}

}

void FloatBuffer_ScalarMul(float* left, float right, int leftCapacity) {

	for (float* leftEnd = left + leftCapacity; left < leftEnd; left++)
		*left *= right;
	
}

void FloatBuffer_Div(float* left, float* right, int leftCapacity, int rightCapacity) {
	
	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left /= *rightHead;
			left++;
		}
	}
	
}

void FloatBuffer_ScalarDiv(float* left, float right, int leftCapacity) {
	FloatBuffer_ScalarMul(left, 1.0f / right, leftCapacity);
}

void FloatBuffer_Add(float* left, float* right, int leftCapacity, int rightCapacity) {

	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left += *rightHead;
			left++;
		}
	}

}

void FloatBuffer_ScalarAdd(float* left, float right, int leftCapacity) {

	for (float* leftEnd = left + leftCapacity; left < leftEnd; left++)
		*left += right;

}

void FloatBuffer_Sub(float* left, float* right, int leftCapacity, int rightCapacity) {
	
	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left -= *rightHead;
			left++;
		}
	}
	
}

void FloatBuffer_ScalarSub(float* left, float right, int leftCapacity) {
	
	for (float* leftEnd = left + leftCapacity; left < leftEnd; left++)
		*left -= right;
	
}

void FloatBuffer_CrossEntropyError(float* res, float* left, float* right, int leftCapacity) {
	
	float sum = 0.0f;
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		sum -= *right * logf(*left + 0.000001f);
		right++;
	}

	*res = sum;
}
	
void FloatBuffer_SoftMax(float* res, float* left, int leftHeight, int leftWidth) {

	int leftCapacity = leftHeight * leftWidth;
	for (int offset = 0; offset < leftCapacity; offset += leftWidth) {

		// MAXを検索
		float* src = left + offset;
		float* dst = res + offset;
		float maxVal = 0.0f;

		{
			float* head = src;
			float* headEnd = head + leftWidth;
			float maxVal = *head;
			for (head++; head < headEnd; head++) {
				float val = *head;
				if (val > maxVal)
					maxVal = val;
			}
		}
		
		// 各要素に exp(a - max) を入れて、ついでに sum(exp(a - max)) を計算
		float sumVal = 0.0f;
		{
			float* src1 = src;
			float* src1End = src1 + leftWidth;
			float* dst1 = dst;
			for (; src1 < src1End; src1++) {
				float val = expf(*src1 - maxVal);
				*dst1 = val;
				sumVal += val;
				dst1++;
			}
		}
		
		// 各要素に 1 / sum(exp(a - max)) を掛け算
		{
			float invSumVal = 1.0f / sumVal;
			float* head = dst;
			float* headEnd = head + leftWidth;
			for (; head < headEnd; head++) {
				*head *= invSumVal;
			}
		}
	}
}

void FloatBuffer_Transpose(float* res, float* left, int leftHeight, int leftWidth) {

	float* leftEnd = left + leftWidth;
	for (; left < leftEnd; left++) {
		float* leftHead = left;
		float* leftHeadEnd = leftHead + (leftHeight * leftWidth);
		for (; leftHead < leftHeadEnd; leftHead += leftWidth) {
			*res = *leftHead;
			res++;
		}
	}
	
}
	
//
void FloatBuffer_ResetZeroOrNegativeAndMakeMask(float* res, float* mask, float* left, int leftCapacity) {

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

void FloatBuffer_ResetZeroOrNegative(float* res, float* left, int leftCapacity) {

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
	
void FloatBuffer_ApplyMask(float* left, float* mask, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		if (*(int*)mask == 0) {
			*(int*)left = 0;
		}
		mask++;
	}
	
}
	
void FloatBuffer_SumToFirstAxis(float* res, float* left, int leftHeight, int leftWidth) {
	
	float* resEnd = res + leftWidth;
	float* leftHeadEnd = left + (leftHeight * leftWidth);
	for (; res < resEnd; res++) {

		float sum = 0.0f;
		for (float* leftHead = left; leftHead < leftHeadEnd; leftHead += leftWidth) {
			sum += *leftHead;
		}
		
		*res = sum;
		left++;
	}
	
}
	
void FloatBuffer_Sqrt(float* left, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		*left = sqrtf(*left);
	}
	
}


} // extern "C"
