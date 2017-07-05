//
//  FloatBufferCpp.hpp
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/08.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#ifndef FloatBufferCpp_hpp
#define FloatBufferCpp_hpp

// Fill buffer with 0.0f.
void _FloatBuffer_FillZero(float* left, int leftCapacity);
// Fill buffer with random values, gaussian distribution.
void _FloatBuffer_FillRandomGaussian(float* left, int leftCapacity);

// TODO: '_'を先頭につける

// Matrix multiplication.
void _FloatBuffer_MatMul(float* res, float* left, float* right, int leftRows, int leftColumns, int rightColumns);


float _FloatBuffer_DotProduct(float* left, float* right, int leftColumns);
void FloatBuffer_Mul(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarMul(float* left, float right, int leftCapacity);
void FloatBuffer_Div(float* left, float* right, int leftCapacity, int rightCapacity);
void FloatBuffer_Add(float* left, float* right, int leftCapacity, int rightCapacity);
void _FloatBuffer_AddScaled(float* left, float* right, float rightScale, int leftCapacity, int rightCapacity);
void FloatBuffer_ScalarAdd(float* left, float right, int leftCapacity);
void FloatBuffer_Sub(float* left, float* right, int leftCapacity, int rightCapacity);
float _FloatBuffer_CrossEntropyError(float* left, float* right, int leftCapacity);
void _FloatBuffer_Softmax(float* res, float* left, int leftRows, int leftColumns);
void _FloatBuffer_Transpose(float* res, float* left, int leftRows, int leftColumns);
void _FloatBuffer_SumToFirstAxis(float* res, float* left, int leftRows, int leftColumns);
void _FloatBuffer_Sqrt(float* left, int leftCapacity);

// Get the index of absolute max value. leftCapacity must not be 0.
int _FloatBuffer_IndexOfAbsMax(float* left, int leftCapacity);

// Calculate the Euclidian norm of a vector.
float _FloatBuffer_Norm(float* left, int leftCapacity);

// Normalize a vector.
float _FloatBuffer_Normalize(float* left, int leftCapacity);
void _FloatBuffer_SafeNormalize(float* left, int leftCapacity);

// Normalize each row of a matrix.
void _FloatBuffer_NormalizeRows(float* left, int leftRows, int leftColumns);

// Perform dot products between a given vector and each rows of matrix.
// (To get the cosine similarity, provide a matrix with each rows normalized and a normalized vector.)
// res: result buffer of size leftRows containing the dot products.
// left: matrix
// right: vector of size leftColumns
void _FloatBuffer_DotProductByRows(float* res, float* left, int leftRows, int leftColumns, float* right);

#endif /* FloatBufferCpp_hpp */
