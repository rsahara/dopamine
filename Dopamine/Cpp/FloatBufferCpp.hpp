//
//  FloatBufferCpp.hpp
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/08.
//  Copyright © 2017年 Runo. All rights reserved.
//

#ifndef FloatBufferCpp_hpp
#define FloatBufferCpp_hpp

void _FloatBuffer_FillZero(float* res, int length);
void _FloatBuffer_FillRandomGaussian(float* res, int length);

// TODO: '_'を先頭につける

void FloatBuffer_MatMul(float* res, float* left, float* right, int leftHeight, int leftWidth, int rightWidth);
void FloatBuffer_Mul(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarMul(float* left, float right, int leftCapacity);
void FloatBuffer_Div(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarDiv(float* left, float right, int leftCapacity);
void FloatBuffer_Add(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarAdd(float* left, float right, int leftCapacity);
void FloatBuffer_Sub(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarSub(float* left, float right, int leftCapacity);
void FloatBuffer_CrossEntropyError(float* res, float* left, float* right, int leftCapacity);
void FloatBuffer_Softmax(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_Transpose(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_SumToFirstAxis(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_Sqrt(float* left, int leftCapacity);


#endif /* FloatBufferCpp_hpp */
