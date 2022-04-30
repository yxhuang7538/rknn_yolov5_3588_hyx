// 检测
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

#include "global.h"

using namespace std;

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int detection_process(const char *model_name int thread_id)
{
    /*
	model_path : rknn模型位置
	thread_id : 进程号
	*/
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(thread_id, &mask); // 绑定cpu
	
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl; // 绑定失败
	
	cout << "NPU进程" << thread_id << "使用 CPU " << thread_id << endl;

    int status = 0;
    rknn_context ctx;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    int ret;

    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    // rknn设置
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

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

    printf("model input height=%d, width=%d, channel=%d\n", height, width,
           channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    void *resize_buf = malloc(height * width * channel);

    // outputs set
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    while (1)
    {
        // 加载图片
        pair<int, Mat> pairIndexImage;
        mtxQueueInput.lock();
        // 若输入队列为空则不进入NPU_process
		if (queueInput.empty())
		{
			mtxQueueInput.unlock();
			usleep(1000);
			// 如果queue处于空且 bReading不在可读取状态则销毁 跳出
			if (bReading) continue;
			else 
			{
				rknn_destroy(ctx);
				break;
			}
		}

        else
		{
			// 加载图片
			cout << "已缓存的图片数： " << queueInput.size() << endl;
			pairIndexImage = queueInput.front();
			printf("Idx:%d 图在线程%d中开始处理\n", pairIndexImage.first, thread_id);
			queueInput.pop();
			mtxQueueInput.unlock();
		}
        cv::Mat img = pairIndexImage.second.clone();
        cv::cvtColor(pairIndexImage.second, img, cv::COLOR_BGR2RGB); // 色彩空间转换
        img_width = img.cols; // 输入图片的宽和高
        img_height = img.rows;
        cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);
        // resize
        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf, width, height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret)
            {
                printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
                return -1;
            }
        IM_STATUS STATUS = imresize(src, dst);
        //cv::imwrite("resize_input.jpg", resize_img);
        inputs[0].buf = resize_buf;
        rknn_inputs_set(ctx, io_num.n_input, inputs);

        gettimeofday(&start_time, NULL);
        ret = rknn_run(ctx, NULL); // 推理
        gettimeofday(&stop_time, NULL);

        printf("once run use %f ms\n",
           (__get_us(stop_time) - __get_us(start_time)) / 1000);

        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        // 后处理(已加入nms)
        // TODO
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

        // Draw Objects
        char text[256];
        for (int i = 0; i < detect_result_group.count; i++)
        {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            printf("%s @ (%d %d %d %d) %f\n",
                det_result->name,
                det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            rectangle(pairIndexImage.second, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
            putText(pairIndexImage.second, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        printf("[%4d/%4d] : worked/total\n", pairIndexImage.first, Frame_cnt);
        printf("Idx:%d 图在线程%d中处理结束\n", pairIndexImage.first, thread_id);

        // 将检测结果加入show序列
        while(idxDectImage != pairIndexImage.first)
		{
			usleep(1000);
		}
        mtxQueueShow.lock();
		idxDectImage++;
        queueShow.push(pairIndexImage.second);
		mtxQueueShow.unlock();

    }
}