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

#define ENABLE_CBLAS 1

#if ENABLE_CBLAS
#include <Accelerate/Accelerate.h>
#endif

extern "C" {

#include "FloatBufferCpp.hpp"

void _FloatBuffer_FillZero(float* left, int leftCapacity) {
	memset(left, 0, leftCapacity * sizeof (float));
}

void _FloatBuffer_FillRandomGaussian(float* left, int leftCapacity) {
	std::random_device dev;
	std::mt19937 gen(dev());
	std::normal_distribution<float> dist;

	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		*left = dist(gen);
	}
}

void _FloatBuffer_MatMul(float* res, float* left, float* right, int leftHeight, int leftWidth, int rightWidth) {

#if ENABLE_CBLAS
	
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

float _FloatBuffer_DotProduct(float* left, float* right, int leftWidth) {

#if ENABLE_CBLAS
	
	return cblas_sdot(leftWidth, left, 1, right, 1);
	
#else

	float res = 0.0f;
	float* leftEnd = left + leftWidth;
	while (left < leftEnd) {
		res += *left * *right;
		left++;
		right++;
	}
	
	return res;

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

#if ENABLE_BLAS

	cblas_sscal(leftCapacity, right, left, 1);
	
#else
	
	for (float* leftEnd = left + leftCapacity; left < leftEnd; left++)
		*left *= right;

#endif
	
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

void FloatBuffer_Add(float* left, float* right, int leftCapacity, int rightCapacity) {

#if ENABLE_BLAS

	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		cblas_saxpy(rightCapacity, 1.0f, right, 1, left, 1);
		left += rightCapacity;
	}

#else

	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left += *rightHead;
			left++;
		}
	}

#endif

}

void _FloatBuffer_AddScaled(float* left, float* right, float rightScale, int leftCapacity, int rightCapacity) {

#if ENABLE_BLAS

	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		cblas_saxpy(rightCapacity, rightScale, right, 1, left, 1);
		left += rightCapacity;
	}

#else
	
	float* leftEnd = left + leftCapacity;
	while (left < leftEnd) {
		float* rightHead = right;
		float* rightHeadEnd = right + rightCapacity;
		for (; rightHead < rightHeadEnd; rightHead++) {
			*left += *rightHead * rightScale;
			left++;
		}
	}
	
#endif

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

float _FloatBuffer_CrossEntropyError(float* left, float* right, int leftCapacity) {
	
	float sum = 0.0f;
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		sum -= *right * logf(*left + 0.000001f);
		right++;
	}

	return sum;
}
	
void _FloatBuffer_Softmax(float* res, float* left, int leftHeight, int leftWidth) {

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

void _FloatBuffer_Transpose(float* res, float* left, int leftHeight, int leftWidth) {

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
	
void _FloatBuffer_SumToFirstAxis(float* res, float* left, int leftHeight, int leftWidth) {
	
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
	
void _FloatBuffer_Sqrt(float* left, int leftCapacity) {
	
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		*left = sqrtf(*left);
	}
	
}

float _FloatBuffer_Norm(float* left, int leftCapacity) {

#if ENABLE_BLAS
	
	return cblas_snrm2(leftCapacity, left, 1);
	
#else
	
	float norm = 0.0f;
	float* leftEnd = left + leftCapacity;
	for (; left < leftEnd; left++) {
		norm += *left * *left;
	}
	return sqrtf(norm);
	
#endif
}

float _FloatBuffer_Normalize(float* left, int leftCapacity) {

	float norm = _FloatBuffer_Norm(left, leftCapacity);
	
	if (norm == 0.0f) {
		left[0] = 1.0f;
	} else {
		FloatBuffer_ScalarMul(left, 1.0f / norm, leftCapacity);
	}
	
	return norm;
}

void _FloatBuffer_NormalizeRows(float* left, int leftRows, int leftColumns) {
	
	for (float* leftEnd = left + (leftRows * leftColumns); left < leftEnd; left += leftColumns) {
		_FloatBuffer_Normalize(left, leftColumns);
	}
	
}

void _FloatBuffer_DotProductByRows(float* res, float* left, int leftRows, int leftColumns, float* right) {

	for (float* resEnd = res + leftRows; res < resEnd; res++)
	{
		*res = _FloatBuffer_DotProduct(left, right, leftColumns);
		left += leftColumns;
	}
}


} // extern "C"
