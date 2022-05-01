// 检测
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

using namespace std;

static void check_ret(int ret, string ret_name)
{
    // 检查ret是否正确并输出，ret_name表示哪一步
    if (ret < 0)
    {
        cout << ret_name << " error ret=" << ret << endl;
    }

}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    // 打印模型输入和输出的信息
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// t为结构体，存储了时间信息：1、tv_sec 代表多少秒；2、tv_usec 代表多少微秒， 1000000 微秒 = 1秒
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec);} 

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    /* 
    加载rknn模型
    filename : rknn模型文件路径
    model_size : 模型的大小
    */
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open rknn model file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int detection_process(const char *model_name, int thread_id, int cpuid)
{
    /*
	model_path : rknn模型位置
	thread_id : 进程号
    cpuid : 使用的cpu
	*/

    /********************初始参数*********************/
    struct timeval start_time, stop_time; // 用于计时
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    /********************绑定cpu*********************/
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask); // 绑定cpu
	
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl; // 绑定失败
	
	cout << "NPU进程" << thread_id << "使用 CPU " << cpuid << endl;

    /********************rknn init*********************/
    string ret_name;
    ret_name = "rknn_init"; // 表示rknn的步骤名称
    rknn_context ctx; // 创建rknn_context对象
    int model_data_size = 0; // 模型的大小
    unsigned char *model_data = load_model(model_name, &model_data_size); // 加载RKNN模型
    /* 初始化参数flag
    RKNN_FLAG_COLLECT_PERF_MASK：用于运行时查询网络各层时间。
    RKNN_FLAG_MEM_ALLOC_OUTSIDE：用于表示模型输入、输出、权重、中间 tensor 内存全部由用户分配。
    */
    int ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL); // 初始化RKNN
    check_ret(ret, ret_name);
    // 设置NPU核心为自动调度
    rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
    ret = rknn_set_core_mask(ctx, core_mask);

    /********************rknn query*********************/
    // rknn_query 函数能够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
    // SDK 版本、内存占用信息、用户自定义字符串等信息。
    // 版本信息
    ret_name = "rknn_query";
    rknn_sdk_version version; // SDK版本信息结构体
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    check_ret(ret, ret_name);
    printf("sdk api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);

    // 输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    check_ret(ret, ret_name);
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 输入输出Tensor属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs)); // 初始化内存
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i; // 输入的索引位置
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 模型输入信息
    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        height = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    multi_npu_process_initialized[thread_id] = 1; // 进程设置完成标志置1
    /********************update frame(user)*********************/
    while(1)
    {
        // 从输入队列加载图片
        pair<int, cv::Mat> frame; // <图的id，图>
        mtxQueueInput.lock();
        // 如果queue处于空且 bReading 不在可读取状态则销毁跳出
        if (queueInput.empty())
		{
			mtxQueueInput.unlock();
			usleep(1000);
			if (bReading) continue;
			else 
			{   
                // 释放内存rknn_context
                ret_name = "rknn_destroy";
				int ret = rknn_destroy(ctx);
				check_ret(ret, ret_name);
                break;
			}
		}

        // 读取到了图片，进行检测
        else
        {
            // 加载图片
			cout << "已缓存的图片数： " << queueInput.size() << endl;
			frame = queueInput.front();
			printf("Idx:%d 图在线程%d中开始处理\n", frame.first, thread_id);
			queueInput.pop();
			mtxQueueInput.unlock();

            cv::Mat img = frame.second.clone();
            cv::cvtColor(frame.second, img, cv::COLOR_BGR2RGB); // 色彩空间转换
            img_width = img.cols; // 输入图片的宽、高和通道数
            img_height = img.rows;
            img_channel = 3;
            // Resize (TODO)

            /********************rknn inputs set*********************/
            ret_name = "rknn_inputs_set";
            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
    
            inputs[0].index = 0; // 输入的索引位置
            inputs[0].type = RKNN_TENSOR_UINT8; // 输入数据类型 采用INT8
            inputs[0].size = width * height * channel; // 这里用的是模型的
            inputs[0].fmt = input_attrs[0].fmt; // 输入格式，NHWC
            inputs[0].pass_through = 0; // 为0代表需要进行预处理
            inputs[0].buf = img.data; // 未进行resize，进行resize需要改为resize的data

            ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
            check_ret(ret, ret_name);

            /********************rknn run****************************/
            ret_name = "rknn_run";
            gettimeofday(&start_time, NULL);
            ret = rknn_run(ctx, NULL); // 推理
            gettimeofday(&stop_time, NULL);
            check_ret(ret, ret_name);
            printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

            /********************rknn outputs get****************************/
            ret_name = "rknn_outputs_get";
            float out_scales[3] = {0, 0, 0}; // 存储scales 和 zp
            int32_t out_zps[3] = {0, 0, 0};
            // 创建rknn_output对象
            rknn_output outputs[io_num.n_output];
            memset(outputs, 0, sizeof(outputs));
            for (int i = 0; i < io_num.n_output; i++) 
            { 
                outputs[i].index = i; // 输出索引
                outputs[i].is_prealloc = 0; // 由rknn来分配输出的buf，指向输出数据
                outputs[i].want_float = 0;
                out_scales[i] = output_attrs[i].scale;
                out_zps[i] = output_attrs[i].zp; 
            }
            ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

            /********************是否打印推理时间细节****************************/
            ret_name = "rknn_perf_detail_display";
            rknn_perf_detail perf_detail;
            ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
            check_ret(ret, ret_name);
            //printf("%s\n",perf_detail.perf_data);

            /********************postprocess_cpu****************************/
            float scale_w = (float)width / img_width; // 图片缩放尺度 resize需要
            float scale_h = (float)height / img_height;
            detect_result_group_t detect_result_group;
            post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

            // 绘制目标检测结果到原frame
            char text[256];
            for (int i = 0; i < detect_result_group.count; i++)
            {
                detect_result_t *det_result = &(detect_result_group.results[i]);
                //sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
                //printf("%s @ (%d %d %d %d) %f\n",
                //    det_result->name,
                //    det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                //    det_result->prop);
                int x1 = det_result->box.left;
                int y1 = det_result->box.top;
                int x2 = det_result->box.right;
                int y2 = det_result->box.bottom;
                int color = det_result->color;
                rectangle(frame.second, cv::Point(x1, y1), cv::Point(x2, y2), randColor[color], 3);
                putText(frame.second, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }

            printf("[%4d/%4d] : worked/total\n", frame.first, Frame_cnt);
            printf("Idx:%d 图在线程%d中处理结束\n", frame.first, thread_id);

            // 将检测结果加入quequeShow队列进行展示或保存为视频
            while(idxDectImage != frame.first) usleep(1000); // 避免多个进程冲突，保证检测顺序正确
            mtxQueueShow.lock();
            idxDectImage++;
            queueShow.push(frame.second);
            mtxQueueShow.unlock();
            if (idxShowImage == Frame_cnt || cv::waitKey(1) == 27)
            {
                cv::destroyAllWindows();
                bReading = false;
                bWriting = false;
                break;
		    }
        } 
    }

    /********************rknn_destroy****************************/
    ret_name = "rknn_destroy";
    ret = rknn_destroy(ctx);
    check_ret(ret, ret_name);

    if (model_data) free(model_data);
    // if (resize_buf) free(resize_buf);
    return 0;
}
