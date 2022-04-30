
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"

#include "global.h"

void videoRead(const char *video_path, int cpuid)
{
	/*
	video_path : 视频路径
	cpuid : 视频读取使用的cpu号
	*/
	
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask); // 绑定cpu
	
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl; // 绑定失败
	
	cout << "视频读取使用 CPU " << cpuid << endl;
	
	cv::VideoCapture video;
	video.open(video_path);
	if (!video.open(video_path)) 
	{
		cout << "Fail to open " << video_path << endl;
		return;
	}
	
	// 获取视频参数
	Frame_cnt = video.get(CAP_PROP_FRAME_COUNT);
    Fps = video.get(CAP_PROP_FPS);
    Video_width = video.get(CAP_PROP_FRAME_WIDTH);
    Video_height = video.get(CAP_PROP_FRAME_HEIGHT);
	
	// 等待npu_process全部完成
	while(1)
	{
		int initialization_finished = 1; // npu_process完成标志
		for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++)
		{
			if (multi_npu_process_initialized[i] == 0)
			{
				initialization_finished = 0 // 标志置0 表示未完成
			}
		}
		
		if (initialization_finished == 1)
		{
			break;
		}
		
	}
	
	// npu_process全部完成
	
	while(1)
	{
		usleep(10);
		Mat img;
		
		if (queueInput.size() < 30)
		{
			// 如果读不到图片，或者bReading不在读取状态，或者读取图像的id大于总帧数则退出读视频
			if (!bReading || !video.read(img) || idxInputImage >= Frame_cnt)
			{
				cout << "读取视频出错！" << endl;
				video.release();
				break;
			}
			// 否则将读取到的图片放入输入队列
			mtxQueueInput.lock();
			queueInput.push(make_pair(idxInputImage++, img));
			mtxQueueInput.unlock();
		}
	}
	
	bReading = false;
	cout << "读取视频结束！" << endl;
}