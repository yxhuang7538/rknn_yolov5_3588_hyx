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
	Frame_cnt = video.get(cv::CAP_PROP_FRAME_COUNT);
    Fps = video.get(cv::CAP_PROP_FPS);
    Video_width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    Video_height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
	
	// 等待npu_process全部完成
	while(1)
	{
		int initialization_finished = 1; // npu_process完成标志
		for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++)
			if (multi_npu_process_initialized[i] == 0) initialization_finished = 0;
		
		if (initialization_finished == 1) break;
	}
	
	// npu_process全部完成
	//cv::Mat img1 = cv::imread("infrared_640.jpg", 1);
	while(1)
	{
		usleep(10);
		cv::Mat img;
		if (queueInput.size() < 30)
		{
			// 如果读不到图片，或者bReading不在读取状态，或者读取图像的id大于总帧数则退出读视频
			if (!bReading || !video.read(img) || idxInputImage >= Frame_cnt)
			{
				cout << "读取视频结束！" << endl;
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

void videoWrite(int cpuid)
{
	/*
	cpuid : 结果视频生成使用的cpu号
	*/
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask); // 绑定cpu
	
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl; // 绑定失败
	
	cout << "写入视频使用 CPU " << cpuid << endl;
	
	cv::VideoWriter vid_writer;
	
	while(1)
	{
		if (Video_width != 0)
		{
			vid_writer  = cv::VideoWriter(SAVE_PATH, cv::VideoWriter::fourcc('M','P','E','G'), Fps, cv::Size(Video_width, Video_height));
			break;
		}
	}
	while(1)
	{
		usleep(100);
		cv::Mat img;
		
		// 如果输出队列存在元素，就一直写入视频
		if (queueOutput.size() > 0) {
			mtxQueueOutput.lock();
			img = queueOutput.front();
			queueOutput.pop();
			mtxQueueOutput.unlock();
			vid_writer.write(img); // Save-video
		}
		
		/*
		if (queueShow.size() > 0)
		{	
			// 目前用来做检测
			mtxQueueShow.lock();
			cv::Mat img = queueShow.front();
			//imshow("RK3588", img);
			queueShow.pop();
			mtxQueueShow.unlock();
			//vid_writer.write(img); // Save-video
			idxShowImage++;

		}
		if (idxShowImage == Frame_cnt || cv::waitKey(1) == 27)
  		{
		    	printf("*******************************************");
			cv::destroyAllWindows();
			bReading = false;
			bWriting = false;
			break;
	    	}
		*/
		if(!bWriting)
		{
			//vid_writer.release();
			break;
		}
	}
	cout << "视频生成！" << endl;
}
