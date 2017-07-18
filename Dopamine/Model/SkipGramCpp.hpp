//
//  SkipGramCpp.hpp
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/31.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#ifndef SkipGramCpp_hpp
#define SkipGramCpp_hpp

// Global initialization.
void _SkipGram_GlobalInit();

// Initialization for a given corpus.
//   itemSequenceBuffer: sequences of item IDs as input. Each sequence of item ID must be terminated with -1. Item IDs start from 0 and are consecutive integers.
//   itemSequenceBufferLength: length of itemSequenceBuffer
//   itemSequenceOffsetArray: [out] array of offset of each sequence.
//   itemSequencesCount:
//     [in] length of itemSequenceOffsetArray, or max. number of sequences to process.
//     [out] number of sequences that were processed.
//   itemNegLotteryInfoArray: [out] information required to perform negative lottery for each item.
//   itemsCount:
//     [in] length of itemNegLotteryInfoArray, or max. number of distinct item ID to process (i.e. max. item ID + 1)
//     [out] number of distinct item IDs actually processed.
void _SkipGram_TrainInit(int* itemSequenceBuffer, int itemSequenceBufferLength, int* itemSequenceOffsetArray, int* itemSequencesCount, float* itemNegLotteryInfoArray, int* itemsCount);

// Training iteration.
//   itemSequenceBuffer: sequences of item IDs as input. Each sequence of item ID must be terminated with -1. Item IDs start from 0 and are consecutive integers.
//   itemSequenceOffsetArray: array of offset of each sequence.
//   itemSequencesCount: number of sequences to process.
//   itemNegLotteryInfoArray: information required to perform negative lottery for each item.
//   itemsCount: number of distinct item IDs to process (i,e, max. item ID + 1). The length of itemNegLotteryInfoArray must be at least itemsCount.
//   itemVectorSize: size of item vectors to train.
//   weightBuffer: [in, out] buffer of all item vectors. Size of this buffer must be (itemsCount * itemVectorSize).
//   negWeightBuffer: [in, out] buffer used to perform negative sampling. Size of this buffer must be (itemsCount * itemVectorSize).
//   tempItemVector: [in, out] temporary buffer required to run the algorithm, of size itemVectorSize. (The returned buffer is meaningless.)
//   windowSize: size of the window. The algorithm will process (windowSize * 2) neighbor items per item.
//   negativeSamplingCount: number of negative sampling per item.
//   learningRate: learning rate (constant for now)
//   iterationsCount: number of iterations to run.
void _SkipGram_TrainIterate(int* itemSequenceBuffer, int* itemSequenceOffsetArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int itemsCount, int itemVectorSize, float* weightBuffer, float* negWeightBuffer, float* tempItemVector, int windowSize, int negativeSamplingCount, float learningRate, int iterationsCount);

#endif /* SkipGramCpp_hpp */
