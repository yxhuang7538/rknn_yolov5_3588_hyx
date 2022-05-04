// 跟踪
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <atomic>
#include <thread>
#include <sys/time.h>
#include <sys/stat.h> // 获取文件属性
#include <dirent.h> // 文件夹操作
#include <unistd.h>
#include <dlfcn.h>
#include "rknn_api.h"

#include "im2d.h"
//#include "RgaUtils.h"
//#include "rga.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"

#include "global.h"
#include "KalmanTracker.h"
#include "Hungarian.h"

using namespace std;

double box_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void track_process(int cpuid)
{
    cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;
	printf("Bind Track process to CPU %d\n", cpuid);

	vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // 跟踪目标id，在初始时置为0

	vector<detect_result_t> frameTrackingResult; // 每帧的跟踪结果
	cout << "跟踪初始化结束！" << endl;

	while (1)
	{
		mtxQueueShow.lock();
		if (queueShow.empty())
		{
			mtxQueueShow.unlock();
			usleep(1000); // 等待检测结果
		}

		// 如果下一个要显示的图片已经被处理好了 则进行追踪任务
		else if (idxShowImage == queueShow.top().frame_id)
		{
			cout << "待追踪的图片数： " << queueShow.size() << endl;
			detect_result_group_t group = queueShow.top(); // 取出图片检测结果group
			cv::Mat img = group.img.clone();
			queueShow.pop();
			mtxQueueShow.unlock();
			if (trackers.size() == 0) // 追踪器是空（第一帧 或者 跟踪目标丢失）
			{
				//用第一帧的检测结果初始化跟踪器
				for (int i = 0; i < group.count; i++)
				{	
					detect_result_t *det_result = &(group.results[i]);
					KalmanTracker trk = KalmanTracker(det_result->box.bbox);
					trackers.push_back(trk);
				}
			}
			else
			{
				// 预测已有跟踪器在当前帧的bbox
				vector<cv::Rect_<float>> predictedBoxes;
				for (auto it = trackers.begin(); it != trackers.end();)
				{
					cv::Rect_<float> pBox = (*it).predict();
					if (pBox.x >= 0 && pBox.y >= 0)
					{
						predictedBoxes.push_back(pBox);
						it++;
					}
					else it = trackers.erase(it); //bb不合理的tracker会被清除
				}

				// 匈牙利算法进行匹配
				vector<vector<double>> iouMatrix;
				iouMatrix.clear();
				unsigned int trkNum = 0;
				unsigned int detNum = 0;
				trkNum = predictedBoxes.size(); //由上一帧预测出来的结果
				detNum = group.count; //当前帧的所有检测结果的 视作传感器的结果
				iouMatrix.resize(trkNum, vector<double>(detNum, 0)); //提前开好空间 避免频繁重定位
				for (unsigned int i = 0; i < trkNum; i++) // 计算IOU矩阵
				{
					for (unsigned int j = 0; j < detNum; j++)
					{
						iouMatrix[i][j] = 1 - box_iou(predictedBoxes[i], group.results[j].box.bbox);
					}
				}

				HungarianAlgorithm HungAlgo;
				vector<int> assignment; //匹配结果 给每一个trk找一个det
				assignment.clear();
                if(trkNum!=0) HungAlgo.Solve(iouMatrix, assignment);//匈牙利算法核心

				// 寻找匹配 未匹配检测 未匹配预测
				set<int> unmatchedDetections; // 没有被配对的检测框 说明有新目标出现
				set<int> unmatchedTrajectories; // 没有被配对的追踪器 说明有目标消失
				set<int> allItems;
				set<int> matchedItems;
				vector<cv::Point> matchedPairs; // 最终配对结果 trk-det
				unmatchedTrajectories.clear();
				unmatchedDetections.clear();
				allItems.clear();
				matchedItems.clear();

				if (detNum > trkNum) // 检测框的数量 大于 现存追踪器的数量
				{
					for (unsigned int n = 0; n < detNum; n++) allItems.insert(n);
					for (unsigned int i = 0; i < trkNum; ++i) matchedItems.insert(assignment[i]);
					/*
					set_difference, 求集合1与集合2的差集 即可以找到没有被追踪的 det
					参数：第一个集合的开始位置，第一个集合的结束位置，
					第二个参数的开始位置，第二个参数的结束位置，
					结果集合的插入迭代器。
					*/
					set_difference( 
						allItems.begin(), allItems.end(),
						matchedItems.begin(), matchedItems.end(),
						insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin())
					);
				}

				else if (detNum < trkNum) // 检测框的数量 小于 现存追踪器的数量; 追踪目标暂时消失
				{
					for (unsigned int i = 0; i < trkNum; ++i)
						if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
							unmatchedTrajectories.insert(i);
				}

				else; // 两者数量相等不做操作

				// 过滤掉低IOU的匹配
				matchedPairs.clear();
				for (unsigned int i = 0; i < trkNum; ++i)
				{
					if (assignment[i] == -1) continue;
					if (1 - iouMatrix[i][assignment[i]] < IOU_THRESH)
					{
						unmatchedTrajectories.insert(i);
						unmatchedDetections.insert(assignment[i]);
					}
					else
						matchedPairs.push_back(cv::Point(i, assignment[i])); // 符合条件 成功配对
				}

				// 更新跟踪器
				int detIdx, trkIdx;
				for (unsigned int i = 0; i < matchedPairs.size(); i++)
				{
					trkIdx = matchedPairs[i].x;
					detIdx = matchedPairs[i].y;
					trackers[trkIdx].update(group.results[detIdx].box.bbox);
				}

				// 给未匹配到的检测框创建和初始化跟踪器
				// unmatchedTrajectories没有操作 所以有必要保存unmatchedTrajectories吗?(maybe not)
				for (auto umd : unmatchedDetections)
				{
					KalmanTracker tracker = KalmanTracker(group.results[umd].box.bbox);
					trackers.push_back(tracker);
				}
			}

			// 获得跟踪器输出
			int max_age = 1;
			int min_hits = 3;
			//m_time_since_update：tracker距离上次匹配成功间隔的帧数
			//m_hit_streak：tracker连续匹配成功的帧数
			frameTrackingResult.clear();
			for (auto it = trackers.begin(); it != trackers.end();)
			{
				// 输出条件：当前帧和前面2帧（连续3帧）匹配成功才记录
				if (((*it).m_time_since_update < 1) &&
					((*it).m_hit_streak >= min_hits || idxShowImage <= min_hits))//河狸
				{
					detect_result_t res;
					res.box.bbox = (*it).get_state();
					res.track_id = (*it).m_id + 1;
					frameTrackingResult.push_back(res);
					it++;
				}
				else
					it++;
				if (it != trackers.end() && (*it).m_time_since_update > max_age)//连续3帧还没匹配到，清除
					it = trackers.erase(it);
			}

			// 绘图
			for (auto tb : frameTrackingResult)
			{
				cv::rectangle(img, tb.box.bbox, randColor[tb.track_id % COLORS_NUMBER], 2, 8, 0);
				cv::putText(img, "Id:"+to_string(tb.track_id), cv::Point(tb.box.bbox.x, tb.box.bbox.y), 1, 1, randColor[tb.track_id % COLORS_NUMBER], 1);
			}

			imshow("rk3588",img);
			mtxQueueOutput.lock();
			queueOutput.push(img);
			mtxQueueOutput.unlock();
			idxShowImage++;

			// 因为此时一定允许过至少一次videoRead 因此frame_cnt一定不为0
			if (idxShowImage == Frame_cnt || cv::waitKey(1) == 27) {
				cv::destroyAllWindows();
				bReading = false;
				bWriting = false;
				break;
			}
		}

		else mtxQueueShow.unlock();
	}
	
}
