//
//  SkipGramCpp.cpp
//  dopamine
//
//  Created by 佐原 瑠能 on 2017/05/31.
//  Copyright © 2017年 Runo. All rights reserved.
//

#include <cstring>
#include <cmath>
#include <random>

extern "C" {
	
#include "FloatBufferCpp.hpp"
#include "SkipGramCpp.hpp"

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

	// バッファーを列挙する
	int* itemSequenceHead = itemSequenceBuffer;
	int* itemSequenceBufferEnd = itemSequenceBuffer + itemSequenceBufferLength;
	for (; itemSequenceHead < itemSequenceBufferEnd; itemSequenceHead++) {

		int itemIndex = *itemSequenceHead;
		
		// シーケンスの終了チェック
		if (itemIndex < 0) {
			
			// 最後のシーケンス
			if (itemSequenceHead + 1 == itemSequenceBufferEnd
				|| maxItemSequencesCount == curItemSequenceIndex) {
				break;
			}
			
			itemSequenceOffsetArray[curItemSequenceIndex] = (int)(itemSequenceHead - itemSequenceBuffer) + 1;
			curItemSequenceIndex++;
		}
		
		// アイテムIDの上限チェック
		if (itemIndex >= maxItemsCount) {
			break;
		}
		
		// 最大値更新
		if (itemIndex > maxItemIndex) {
			maxItemIndex = itemIndex;
		}
		
		// アイテムをカウント
		itemNegLotteryInfoArray[itemIndex] += 1.0f;
	}

	// 結果を設定
	*itemSequencesCount = curItemSequenceIndex;
	int actualItemsCount = maxItemIndex + 1;
	*itemsCount = actualItemsCount;

	// 抽選データを準備
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

	return maxIndex % itemsCount; // TODO: 整理、デバッグ
}

void _SkipGram_TrainIterate(int* itemSequenceBuffer, int* itemSequenceOffsetArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int itemsCount, int itemVectorSize, float* weightBuffer, float* negWeightBuffer, float* tempItemVector, int windowSize, int negativeSamplingCount, float learningRate) {
	
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (int itemSequenceIndex = 0; itemSequenceIndex < itemSequencesCount; itemSequenceIndex++) { // for each sequence
		
		int* itemSequenceStart = itemSequenceBuffer + itemSequenceOffsetArray[itemSequenceIndex];
		for (int* headItem = itemSequenceStart; ; headItem++) { // for each item
			int itemIndex = *headItem;
			if (itemIndex < 0) {
				break; // End of sequence.
			}
			
			int* headRelated = headItem - windowSize;
			int* headRelatedEnd = headItem + windowSize;
			if (headRelated < itemSequenceStart) {
				headRelated = itemSequenceStart;
			}
			
			for (; headRelated <= headRelatedEnd; headRelated++) { // for each related item
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
				for (int negativeSamplingIndex = -1; negativeSamplingIndex < negativeSamplingCount; negativeSamplingIndex++) { // for each (positive/negative) sampling
					
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
					float expDotProduct = expf(dotProduct); // TODO: table lookup
					float delta = (label - (expDotProduct / (expDotProduct + 1.0f))) * learningRate;
					
#if 0
					_FloatBuffer_AddScaled(tempItemVector, targetVector, delta, itemVectorSize, itemVectorSize);
					_FloatBuffer_AddScaled(targetVector, relatedVector, delta, itemVectorSize, itemVectorSize);
#else
					for (int featureIndex = 0; featureIndex < itemVectorSize; featureIndex++) {
						tempItemVector[featureIndex] += delta * targetVector[featureIndex];
						targetVector[featureIndex] += delta * relatedVector[featureIndex];
					}
#endif
					
				} // for each (positive/negative) sampling
				
				FloatBuffer_Add(relatedVector, tempItemVector, itemVectorSize, itemVectorSize);
				
			} // for each related item

		} // for each item

	} // for each sequence

}

}
