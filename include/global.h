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

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     2
#define NMS_THRESH        0.6
#define BOX_THRESH        0.5
#define IOU_THRESH        0.3 // 跟踪用
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)
#define LABEL_NALE_TXT_PATH "./model/labels.txt"
#define SAVE_PATH "output.avi"
#define COLORS_NUMBER     20 // 20个随机颜色

using namespace std;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
    cv::Rect_<float> bbox;
} BOX_RECT; // box格式 左上 右下 点坐标

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE]; // 物体类别名字
    BOX_RECT box; // 目标box
    float prop; // 类别概率
    int color; // 目标对应类别的颜色
    int track_id; // 跟踪的时候确定目标实例id
} detect_result_t;

typedef struct _detect_result_group_t // 多个检测结果组
{
    int id; // 类别id
    int count; // 一张图框的总数
    int frame_id; // 第几帧
    cv::Mat img; // 原图
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

class paircomp {
public:
    bool operator()(const detect_result_group_t &n1, const detect_result_group_t &n2) const {
        return n1.frame_id > n2.frame_id;
    }
};

extern mutex mtxQueueInput; // 输入队列mutex
extern mutex mtxQueueOutput; // 输出队列mutex
extern mutex mtxQueueShow; // 展示队列mutex
extern queue<pair<int, cv::Mat>> queueInput; // 输入队列 <id, 图片>
extern queue<cv::Mat> queueOutput; // 输出队列 <图片>
extern priority_queue<detect_result_group_t, vector<detect_result_group_t>, paircomp> queueShow;
extern int Frame_cnt; // 帧的计数
extern int Fps; // 帧率
extern int Video_width; // 视频的输入宽度
extern int Video_height; // 视频的输入高度

extern int multi_npu_process_initialized[4]; // npu初始化完成标志，1为完成，0为未完成

extern int idxInputImage; // 输入视频的帧的id
extern int idxDectImage; // 要检测的下一帧id
extern int idxShowImage; // 要显示的下一帧的id
extern bool bReading;   // flag of input
extern bool bWriting;	// flag of output
extern double Time_video;  // 整个视频(包括画图)所花的时间
extern double Time_track;  // 整个视频追踪所花的时间

extern cv::Scalar_<int> randColor[COLORS_NUMBER]; //随机颜色
extern cv::RNG rng;


int detection_process(const char *model_name, int thread_id, int cpuid);

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 int32_t *qnt_zps, float *qnt_scales,
                 detect_result_group_t *group);
                 
void videoRead(const char *video_path, int cpuid);

void videoWrite(int cpuid);
void track_process(int cpuid);
double __get_us(struct timeval t);
