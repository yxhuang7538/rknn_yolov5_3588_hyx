// 声明所有需要用到的全局变量
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <atomic>
#include <queue>
#include <set>
#include <thread> // 线程库
#include <mutex>
#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <dlfcn.h>
#include "rknn_api.h"

#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "global.h"

mutex mtxQueueInput; // 输入队列mutex
mutex mtxQueueOutput; // 输出队列mutex
mutex mtxQueueShow; // 展示队列mutex
queue<pair<int, cv::Mat>> queueInput; // 输入队列 <id, 图片>
queue<cv::Mat> queueOutput; // 输出队列 <图片>
queue<cv::Mat> queueShow;
int Frame_cnt = 0; // 帧的计数
int Fps = 0; // 帧率
int Video_width = 0; // 视频的输入宽度
int Video_height = 0; // 视频的输入高度

int multi_npu_process_initialized[5] = {0, 0, 0, 0, 0}; // npu初始化完成标志，1为完成，0为未完成

int idxInputImage = 0; // 输入视频的帧的id
int idxDectImage = 0; // 要检测的下一帧id
int idxShowImage = 0; // 要显示的下一帧的id
bool bReading = true;   // flag of input
bool bWriting = true;	// flag of output
double Time_video = 0;  // 整个视频(包括画图)所花的时间
double Time_track = 0;  // 整个视频追踪所花的时间

cv::Scalar_<int> randColor[COLORS_NUMBER]; // 随机颜色
cv::RNG rng(0xFFFFFFFF); //RNG类是opencv里C++的随机数产生器
for (int i = 0; i < COLORS_NUMBER; i++) rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);