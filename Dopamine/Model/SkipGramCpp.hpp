//
//  SkipGramCpp.hpp
//  dopamine
//
//  Created by Runo Sahara on 2017/05/31.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

#ifndef SkipGramCpp_hpp
#define SkipGramCpp_hpp

// 全体の初期化
void _SkipGram_GlobalInit();

// 学習の初期化
// itemSequenceBuffer: 入力とするアイテムIDのシーケンスの配列（アイテムIDは0から始まる値、アイテムのシーケンスはアイテムID-1で終わる）
// itemSequenceBufferLength: itemSequenceBufferの長さ
// itemSequenceOffsetArray: [out] アイテムシーケンス先頭オフセットの配列
// itemSequencesCount:
//   [in] itemSequenceOffsetArrayの長さ、又は扱うシーケンス数の最大数
//   [out] 処理したアイテムのシーケンス数
// itemNegLotteryInfoArray: [out] アイテムごとのネガティブ抽選用の情報配列
// itemsCount:
//   [in] itemNegLotteryInfoArray の長さ、又は扱うアイテムの最大数（＝アイテムIDの最大値+1）
//   [out] 処理したアイテム数
void _SkipGram_TrainInit(int* itemSequenceBuffer, int itemSequenceBufferLength, int* itemSequenceOffsetArray, int* itemSequencesCount, float* itemNegLotteryInfoArray, int* itemsCount);


// 学習のイテレーション
// itemSequenceBuffer: 入力とするアイテムIDのシーケンスの配列（アイテムIDは0から始まる値、アイテムのシーケンスはアイテムID-1で終わる）
// itemSequenceOffsetArray: アイテムシーケンス先頭オフセットの配列
// itemSequencesCount: 扱うアイテムシーケンスの数
// itemNegLotteryInfoArray: アイテムごとのネガティブ抽選用の情報配列
// itemsCount: 有効とするアイテムIDの最大値+1（itemNegLotteryInfoArrayの長さはitemsCount以上）
// itemVectorSize: アイテムの特徴ベクトルのサイズ
// weightBuffer: [in, out] 全アイテムの特徴ベクトルのバッファ、itemsCount x itemVectorSize の行列
// negWeightBuffer: [in, out] 特徴ベクトルネガティブサンプリング計算用のバッファ、itemsCount x itemVectorSize の行列
// tempItemVector: [in, out] 計算に必要な一時メモリ領域、itemVectorSize の長さのバッファー（特に意味のあるデータは出力しない）
// windowSize: 窓の大きさ
// negativeSamplingCount: ネガティブサンプル数
// learningRate: 学習の係数
// iterationsCount: イテレーションの回数
void _SkipGram_TrainIterate(int* itemSequenceBuffer, int* itemSequenceOffsetArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int itemsCount, int itemVectorSize, float* weightBuffer, float* negWeightBuffer, float* tempItemVector, int windowSize, int negativeSamplingCount, float learningRate, int iterationsCount);

#endif /* SkipGramCpp_hpp */
