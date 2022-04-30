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
    array<thread, cpus> threads;

    // 分配进程
    if (argv[1][0] == 'v')
    {
        // 检测视频
        threads = {
            thread(videoRead, argv[3], 5),
            thread(videoWrite, 6),
            thread(track_process, 7),
            thread(detection_process, argv[2], 0),
            thread(detection_process, argv[2], 1),
            thread(detection_process, argv[2], 2),
            thread(detection_process, argv[2], 3),
            thread(detection_process, argv[2], 4)
        }
        for (int i = 0; i < 8; i++) threads[i].join(); // join进程

        // 结果显示 TODO
    }

    else
    {
        // 摄像头检测 TODO
        cout << "摄像头检测" << endl;
    }
}