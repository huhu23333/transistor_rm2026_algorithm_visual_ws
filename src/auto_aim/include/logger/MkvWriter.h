#include <iostream>
#include <vector>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>   // 添加此行
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>

class MkvAllIntraWriter {
public:
    MkvAllIntraWriter() : fmtCtx_(nullptr), codecCtx_(nullptr), swsCtx_(nullptr),
                          videoStream_(nullptr), frame_(nullptr), frameCount_(0) {}

    ~MkvAllIntraWriter() {
        close();
    }

    /**
     * 打开输出文件并初始化编码器
     * @param filename 输出文件名（建议 .mkv）
     * @param width    视频宽度
     * @param height   视频高度
     * @param fps      帧率
     * @param bitrate  编码码率（bps），例如 8000000 表示 8 Mbps
     * @return true 成功，false 失败
     */
    bool open(const std::string& filename, int width, int height, double fps, int64_t bitrate);

    /**
     * 写入一帧 cv::Mat（BGR 格式）
     * @param bgrMat OpenCV Mat，尺寸需与构造时一致
     * @return true 成功，false 失败
     */
    bool writeFrame(const cv::Mat& bgrMat);

    /**
     * 关闭写入器，写入文件尾并释放资源
     */
    void close();

private:
    AVFormatContext* fmtCtx_;
    AVCodecContext*  codecCtx_;
    SwsContext*      swsCtx_;
    AVStream*        videoStream_;
    AVFrame*         frame_;
    int64_t          frameCount_;
};
