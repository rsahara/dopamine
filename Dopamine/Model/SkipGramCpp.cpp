//
//  SkipGramCpp.cpp
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/31.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#include <cstring>
#include <cmath>
#include <random>

extern "C" {
	
#include "FloatBufferCpp.hpp"
#include "SkipGramCpp.hpp"

#define EXPTABLE_SIZE		1000
#define EXPTABLE_INPUTMAX	6.0f

float* _SkipGram_ExpTable = NULL;
	
void _SkipGram_GlobalInit() {
	
	// Calculation table for exp/(exp+1)
	if (_SkipGram_ExpTable == nullptr) {
		_SkipGram_ExpTable = (float*)malloc(EXPTABLE_SIZE * sizeof(float));
		for (int tableIndex = 0; tableIndex < EXPTABLE_SIZE; tableIndex++) {
			float input = (((float)tableIndex) / ((float)EXPTABLE_SIZE)) * (EXPTABLE_INPUTMAX * 2.0f) - EXPTABLE_INPUTMAX;
			float val = expf(input);
			_SkipGram_ExpTable[tableIndex] = val / (val + 1.0f);
		}
	}
	
}

void _SkipGram_TrainInit(int* itemSequenceBuffer, int itemSequenceBufferLength, int* itemSequenceOffsetArray, int* itemSequencesCount, float* itemNegLotteryInfoArray, int* itemsCount) {

	int maxItemSequencesCount = *itemSequencesCount;
	int maxItemsCount = *itemsCount;
	
	if (maxItemSequencesCount == 0 || maxItemsCount == 0) {
		*itemSequencesCount = 0;
		*itemsCount = 0;
		return;
	}
	
	int maxItemIndex = 0;
	memset(itemNegLotteryInfoArray, 0, maxItemsCount * sizeof (float));
	itemSequenceOffsetArray[0] = 0;
	int curItemSequenceIndex = 1;

	// Process buffer.
	int* itemSequenceHead = itemSequenceBuffer;
	int* itemSequenceBufferEnd = itemSequenceBuffer + itemSequenceBufferLength;
	for (; itemSequenceHead < itemSequenceBufferEnd; itemSequenceHead++) {

		int itemIndex = *itemSequenceHead;
		
		// Check end of sequence.
		if (itemIndex < 0) {
			if (itemSequenceHead + 1 == itemSequenceBufferEnd
				|| maxItemSequencesCount == curItemSequenceIndex) {
				break;
			}
			
			itemSequenceOffsetArray[curItemSequenceIndex] = (int)(itemSequenceHead - itemSequenceBuffer) + 1;
			curItemSequenceIndex++;
		}
		
		// Check if we reached the max number of items.
		if (itemIndex >= maxItemsCount) {
			break;
		}
		
		// Update the max item ID.
		if (itemIndex > maxItemIndex) {
			maxItemIndex = itemIndex;
		}
		
		// Count items.
		itemNegLotteryInfoArray[itemIndex] += 1.0f;
	}

	// Set the result.
	*itemSequencesCount = curItemSequenceIndex;
	int actualItemsCount = maxItemIndex + 1;
	*itemsCount = actualItemsCount;

	// Prepare lottery data for negative sampling.
	float sum = 0.0f;
	float* itemNegLotteryInfoEnd = itemNegLotteryInfoArray + actualItemsCount;
	for (float* head = itemNegLotteryInfoArray; head < itemNegLotteryInfoEnd; head++) {
		*head = powf(*head, 0.75f);
		sum += *head;
	}
	float sumInv = 1.0f / sum;
	float acc = 0.0f;
	for (float* head = itemNegLotteryInfoArray; head < itemNegLotteryInfoEnd; head++) {
		acc += *head * sumInv;
		*head = acc;
	}
}

inline int _SkipGram_RandomNegativeItemIndex(float* itemNegLotteryInfoArray, int itemsCount, float randVal) {

	int minIndex = 0;
	int maxIndex = itemsCount - 1;
	while (minIndex < maxIndex) {
		int testIndex = (minIndex + maxIndex) / 2;
		float testVal = itemNegLotteryInfoArray[testIndex];
		if (randVal < testVal) {
			maxIndex = testIndex - 1;
		} else {
			minIndex = testIndex + 1;
		}
	}
	
	if (maxIndex < 0)
		return 0;
	if (maxIndex >= itemsCount)
		return itemsCount - 1;
	return maxIndex;
}
	
inline float _SkipGram_ExpTableOutput(float input) {

	int tableIndex = (input + EXPTABLE_INPUTMAX) * (((float)EXPTABLE_SIZE) / EXPTABLE_INPUTMAX / 2.0f);
	if (tableIndex < 0)
		return 0.0f;
	if (tableIndex >= EXPTABLE_SIZE)
		return 1.0f;
	
	return _SkipGram_ExpTable[tableIndex];
}

void _SkipGram_TrainIterate(int* itemSequenceBuffer, int* itemSequenceOffsetArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int itemsCount, int itemVectorSize, float* weightBuffer, float* negWeightBuffer, float* tempItemVector, int windowSize, int negativeSamplingCount, float learningRate, int iterationsCount) {
	
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_real_distribution<float> dist(0.0f, itemNegLotteryInfoArray[itemsCount - 1]);

	while (iterationsCount--) { // For each iteration.
		for (int itemSequenceIndex = 0; itemSequenceIndex < itemSequencesCount; itemSequenceIndex++) { // For each sequence.
			
			int* itemSequenceStart = itemSequenceBuffer + itemSequenceOffsetArray[itemSequenceIndex];
			for (int* headItem = itemSequenceStart; ; headItem++) { // For each item.
				int itemIndex = *headItem;
				if (itemIndex < 0) {
					break; // End of sequence.
				}
				
				int* headRelated = headItem - windowSize;
				int* headRelatedEnd = headItem + windowSize;
				if (headRelated < itemSequenceStart) {
					headRelated = itemSequenceStart;
				}
				
				for (; headRelated <= headRelatedEnd; headRelated++) { // For each related item.
					int relatedItemIndex = *headRelated;
					if (relatedItemIndex == itemIndex) {
						continue; // Skip center item.
					}
					if (relatedItemIndex < 0) {
						break; // End of sequence.
					}

					memset(tempItemVector, 0, itemVectorSize * sizeof (float));
					
					float* relatedVector = weightBuffer + (itemVectorSize * relatedItemIndex);
					
					// Positive sampling, then negative sampling.
					float label;
					int targetItemIndex;
					for (int negativeSamplingIndex = -1; negativeSamplingIndex < negativeSamplingCount; negativeSamplingIndex++) { // For each (positive/negative) sampling.
						
						if (negativeSamplingIndex == -1) {
							targetItemIndex = itemIndex;
							label = 1.0f;
						} else {
							targetItemIndex = _SkipGram_RandomNegativeItemIndex(itemNegLotteryInfoArray, itemsCount, dist(gen));
							if (targetItemIndex == itemIndex) {
								continue;
							}
							label = 0.0f;
						}
						
						float* targetVector = negWeightBuffer + (itemVectorSize * targetItemIndex);

						float dotProduct = _FloatBuffer_DotProduct(relatedVector, targetVector, itemVectorSize);
						float delta = (label - _SkipGram_ExpTableOutput(dotProduct)) * learningRate;
#if 0 // naive code
						float expDotProduct = expf(dotProduct);
						float delta = (label - (expDotProduct / (expDotProduct + 1.0f))) * learningRate;
#endif

						_FloatBuffer_AddScaled(tempItemVector, targetVector, delta, itemVectorSize, itemVectorSize);
						_FloatBuffer_AddScaled(targetVector, relatedVector, delta, itemVectorSize, itemVectorSize);
						
#if 0 // naive code
						float* headTempItemVector = tempItemVector;
						float* headRelatedVector = relatedVector;
						for (float* targetVectorEnd = targetVector + itemVectorSize; targetVector < targetVectorEnd; targetVector++) {
							*headTempItemVector += delta * *targetVector;
							*targetVector += delta * *headRelatedVector;
							
							headTempItemVector++;
							headRelatedVector++;
						}
#endif

					} // For each (positive/negative) sampling.
					
					_FloatBuffer_Add(relatedVector, tempItemVector, itemVectorSize, itemVectorSize);
					
				} // For each related item.

			} // For each item.

		} // For each sequence.
		
	} // For each iteration.

}

}
