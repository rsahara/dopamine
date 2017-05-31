//
//  SkipGramCpp.hpp
//  dopamine
//
//  Created by 佐原 瑠能 on 2017/05/31.
//  Copyright © 2017年 Runo. All rights reserved.
//

#ifndef SkipGramCpp_hpp
#define SkipGramCpp_hpp

// 初期化
// itemSequenceArray: アイテムIDシーケンスの先頭ポインタの配列（アイテムIDは0から始まる連番、シーケンスは-1で終わる）
// itemSequencesCount: itemSequenceArrayの長さ
// itemNegLotteryInfoArray: [out] アイテムごとのネガティブ抽選用の情報配列
// itemsCount:
//   [in] itemNegLotteryInfoArray の長さ、又は有効とするアイテムIDの最大値+1
//   [out] 実際にカウントしたアイテム数を返す
void _SkipGram_Init(int** itemSequenceArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int* itemsCount);


// 初期化
// itemSequenceArray: アイテムIDシーケンスの先頭ポインタの配列（アイテムIDは0から始まる連番、シーケンスは-1で終わる）
// itemSequencesCount: itemSequenceArrayの長さ
// itemNegLotteryInfoArray: アイテムごとのネガティブ抽選用の情報配列
// itemsCount: 有効とするアイテムIDの最大値+1（itemNegLotteryInfoArrayの長さはitemsCount以上）
// itemVectorSize: アイテムの特徴ベクトルのサイズ
// weightBuffer: [in, out] 全アイテムの特徴ベクトルのバッファ、itemsCount x itemVectorSize の行列
// negWeightBuffer: [in, out] 特徴ベクトルネガティブサンプリング計算用のバッファ、itemsCount x itemVectorSize の行列
// tempItemVector: 計算に必要な一時メモリ領域、itemVectorSize の長さのバッファー
void _SkipGram_Iterate(int** itemSequenceArray, int itemSequencesCount, float* itemNegLotteryInfoArray, int itemsCount, int itemVectorSize, float* weightBuffer, float* negWeightBuffer, float* tempItemVector);

#endif /* SkipGramCpp_hpp */
