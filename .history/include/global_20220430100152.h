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
// #include "rknn_api_1808.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"
#include "KalmanTracker.h"
#include "Hungarian.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.6
#define BOX_THRESH        0.5
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)
#define LABEL_NALE_TXT_PATH "./model/labels.txt"

using namespace std;

static int Frame_cnt = 0; // 帧的计数
static int Fps = 0; // 帧率
static int Video_width = 0; // 视频的输入宽度
static int Video_height = 0; // 视频的输入高度

static int multi_npu_process_initialized[5] = {0, 0, 0, 0, 0}; // npu初始化完成标志，1为完成，0为未完成

mutex mtxQueueInput; // 输入队列mutex
mutex mtxQueueOutput; // 输出队列mutex
mutex mtxQueueShow; // 展示队列mutex
queue<pair<int, cv::Mat>> queueInput; // 输入队列 <id, 图片>
queue<cv::Mat> queueOutput; // 输出队列 <图片>

static int idxInputImage = 0; // 输入视频的帧的id
static int idxDectImage = 0; // 要检测的下一帧id
static int idxShowImage = 0; // 要显示的下一帧的id
static bool bReading = true;   // flag of input
static bool bWriting = true;	// flag of output
static double Time_video = 0;  // 整个视频(包括画图)所花的时间
static double Time_track = 0;  // 整个视频追踪所花的时间

vector<float> out_scales; // 存储scales 和 zp
vector<int32_t> out_zps;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT; // 画图用的box

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

static int detection_process(const char *model_name int thread_id);

static int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);