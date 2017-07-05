//
//  FloatBufferCpp.cpp
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/08.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#include <cmath>
#include <cstring>
#include <random>

#include "Environment.hpp"

#if ENABLE_APPLE_BLAS
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

void _FloatBuffer_MatMul(float* res, float* left, float* right, int leftRows, int leftColumns, int rightColumns) {

#if ENABLE_APPLE_BLAS
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, leftRows, rightColumns, leftColumns, 1.0f, left, leftColumns, right, rightColumns, 0.0, res, rightColumns);

#else
	
	float* leftEnd = left + (leftRows * leftColumns);
	float* rightHeadEnd = right + rightColumns;
	while (left < leftEnd) {
		float* leftHeadEnd = left + leftColumns;

		for (float* rightHeadStart = right; rightHeadStart < rightHeadEnd; rightHeadStart++) {
			
			float temp = 0.0f;

			float* leftHead = left;
			float* rightHead = rightHeadStart;
			
			while (leftHead < leftHeadEnd) {
				temp += *leftHead * *rightHead;
				leftHead++;
				rightHead += rightColumns;
			}
			
			*res = temp;
			res++;
		}
		
		left = leftHeadEnd;
	}
	
#endif

}

float _FloatBuffer_DotProduct(float* left, float* right, int leftColumns) {

#if ENABLE_APPLE_BLAS
	
	return cblas_sdot(leftColumns, left, 1, right, 1);
	
#else

	float res = 0.0f;
	float* leftEnd = left + leftColumns;
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

#if ENABLE_APPLE_BLAS

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

#if ENABLE_APPLE_BLAS

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

#if ENABLE_APPLE_BLAS

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
	
void _FloatBuffer_Softmax(float* res, float* left, int leftRows, int leftColumns) {

	int leftCapacity = leftRows * leftColumns;
	for (int offset = 0; offset < leftCapacity; offset += leftColumns) {

		// MAXを検索
		float* src = left + offset;
		float* dst = res + offset;
		float maxVal = 0.0f;

		{
			float* head = src;
			float* headEnd = head + leftColumns;
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
			float* src1End = src1 + leftColumns;
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
			float* headEnd = head + leftColumns;
			for (; head < headEnd; head++) {
				*head *= invSumVal;
			}
		}
	}
}

void _FloatBuffer_Transpose(float* res, float* left, int leftRows, int leftColumns) {

	float* leftEnd = left + leftColumns;
	for (; left < leftEnd; left++) {
		float* leftHead = left;
		float* leftHeadEnd = leftHead + (leftRows * leftColumns);
		for (; leftHead < leftHeadEnd; leftHead += leftColumns) {
			*res = *leftHead;
			res++;
		}
	}
	
}
	
void _FloatBuffer_SumToFirstAxis(float* res, float* left, int leftRows, int leftColumns) {
	
	float* resEnd = res + leftColumns;
	float* leftHeadEnd = left + (leftRows * leftColumns);
	for (; res < resEnd; res++) {

		float sum = 0.0f;
		for (float* leftHead = left; leftHead < leftHeadEnd; leftHead += leftColumns) {
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

int _FloatBuffer_IndexOfAbsMax(float* left, int leftCapacity) {

	int resIndex = 0;
	float resVal = *left;
	if (resVal < 0.0f) {
		resVal = -resVal;
	}
	
	for (int leftIndex = 1; leftIndex < leftCapacity; leftIndex++) {
		float val = left[leftIndex];
		if (val < 0.0f) {
			val = -val;
		}
		if (val > resVal) {
			resVal = val;
			resIndex = leftIndex;
		}
	}

	return resIndex;
}

float _FloatBuffer_Norm(float* left, int leftCapacity) {

#if ENABLE_APPLE_BLAS
	
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
	
	if (norm != 0.0f) {
		FloatBuffer_ScalarMul(left, 1.0f / norm, leftCapacity);
	}
	
	return norm;
}

void _FloatBuffer_SafeNormalize(float* left, int leftCapacity) {
	
	int indexOfAbsMax = _FloatBuffer_IndexOfAbsMax(left, leftCapacity);
	float absMax = left[indexOfAbsMax];
	if (absMax == 0.0f) {
		return;
	}
	if (absMax < 0.0f) {
		absMax = -absMax;
	}
	
	FloatBuffer_ScalarMul(left, 1.0f / absMax, leftCapacity);
	float norm = _FloatBuffer_Norm(left, leftCapacity);
	if (norm != 0.0f) {
		FloatBuffer_ScalarMul(left, 1.0f / norm, leftCapacity);
	}
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
