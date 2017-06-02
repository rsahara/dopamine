//
//  FloatBufferCpp.hpp
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/08.
//  Copyright © 2017年 Runo. All rights reserved.
//

#ifndef FloatBufferCpp_hpp
#define FloatBufferCpp_hpp

void _FloatBuffer_FillZero(float* left, int leftCapacity);
void _FloatBuffer_FillRandomGaussian(float* left, int leftCapacity);

// TODO: '_'を先頭につける
// TODO: height/width を使わない

void FloatBuffer_MatMul(float* res, float* left, float* right, int leftHeight, int leftWidth, int rightWidth);
float _FloatBuffer_DotProduct(float* left, float* right, int leftWidth);
void FloatBuffer_Mul(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarMul(float* left, float right, int leftCapacity);
void FloatBuffer_Div(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_Add(float* left, float* right, int leftCapacity, int rightCapacity);
void _FloatBuffer_AddScaled(float* left, float* right, float rightScale, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarAdd(float* left, float right, int leftCapacity);
void FloatBuffer_Sub(float* left, float* right, int leftCapacity, int rightCapacity);
float _FloatBuffer_CrossEntropyError(float* left, float* right, int leftCapacity);
void FloatBuffer_Softmax(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_Transpose(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_SumToFirstAxis(float* res, float* left, int leftHeight, int leftWidth);
void FloatBuffer_Sqrt(float* left, int leftCapacity);

// Calculate the Euclidian norm of a vector.
float _FloatBuffer_Norm(float* left, int leftCapacity);

// Normalizes a vector.
float _FloatBuffer_Normalize(float* left, int leftCapacity);

// Normalizes each row of a matrix.
void _FloatBuffer_NormalizeRows(float* left, int leftRows, int leftColumns);

// Performs dot products between a given vector and each rows of matrix.
// (To get the cosine similarity, provide a matrix with each rows normalized and a normalized vector.)
// res: result buffer of size leftRows containing the dot products.
// left: matrix
// right: vector of size leftColumns
void _FloatBuffer_DotProductByRows(float* res, float* left, int leftRows, int leftColumns, float* right);

#endif /* FloatBufferCpp_hpp */
