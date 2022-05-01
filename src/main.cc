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
#include "global.h"

using namespace std;
int main(const int argc, const char **argv)
{
    // 主函数，主要为各个模块分配进程
    int cpus = sysconf(_SC_NPROCESSORS_CONF); // 获取cpu核的数量
    array<thread, 8> threads;
    struct timeval start_time, stop_time;

    // 分配进程
    gettimeofday(&start_time, NULL);
    if (argv[1][0] == 'v')
    {
        // 检测视频
        threads = {
            thread(videoRead, argv[3], 0),
            thread(videoWrite, 1),
            thread(track_process, 2),
            thread(detection_process, argv[2], 0, 3),
            thread(detection_process, argv[2], 1, 4),
            thread(detection_process, argv[2], 2, 5),
            thread(detection_process, argv[2], 3, 6),
            thread(detection_process, argv[2], 4, 7)
        };
        for (int i = 0; i < 8; i++) threads[i].join(); // join进程

        // 结果显示
        gettimeofday(&stop_time, NULL);
        Time_video = (__get_us(stop_time) - __get_us(start_time)) / 1000;
	    cout << "总耗时:" << Time_video << " 总帧数:" << Frame_cnt  <<endl;
	    cout << "平均帧率:" << Frame_cnt / Time_video << endl;
    }

    else
    {
        // 摄像头检测 TODO
        cout << "摄像头检测" << endl;
    }
}
