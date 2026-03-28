#include "logger/MkvWriter.h"

/**
 * 打开输出文件并初始化编码器
 * @param filename 输出文件名（建议 .mkv）
 * @param width    视频宽度
 * @param height   视频高度
 * @param fps      帧率
 * @param bitrate  编码码率（bps），例如 8000000 表示 8 Mbps
 * @return true 成功，false 失败
 */
bool MkvAllIntraWriter::open(const std::string& filename, int width, int height, double fps, int64_t bitrate) {
    int ret;

    // 1. 分配输出上下文
    ret = avformat_alloc_output_context2(&fmtCtx_, nullptr, nullptr, filename.c_str());
    if (ret < 0 || !fmtCtx_) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Could not create output context, error: " << errbuf << std::endl;
        return false;
    }

    // 2. 查找 H.264 编码器
    AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "H.264 encoder not found" << std::endl;
        goto cleanup;
    }

    // 3. 添加视频流
    videoStream_ = avformat_new_stream(fmtCtx_, codec);
    if (!videoStream_) {
        std::cerr << "Could not create video stream" << std::endl;
        goto cleanup;
    }
    videoStream_->id = fmtCtx_->nb_streams - 1;

    // 4. 分配编码器上下文
    codecCtx_ = avcodec_alloc_context3(codec);
    if (!codecCtx_) {
        std::cerr << "Could not allocate codec context" << std::endl;
        goto cleanup;
    }

    // 基本参数
    codecCtx_->width = width;
    codecCtx_->height = height;
    codecCtx_->time_base = {1, static_cast<int>(fps)}; // 帧率时间基
    videoStream_->time_base = codecCtx_->time_base;
    codecCtx_->framerate = {static_cast<int>(fps), 1};

    codecCtx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codecCtx_->bit_rate = bitrate;
    codecCtx_->gop_size = 1;               // 每个帧都是关键帧
    codecCtx_->keyint_min = 1;              // 最小关键帧间隔
    codecCtx_->max_b_frames = 0;             // 禁用 B 帧

    // 设置编码器为全关键帧模式（x264 特定参数）
    av_opt_set(codecCtx_->priv_data, "x264-params", "keyint=1:min-keyint=1", 0);
    // 可选：设置 preset 和 tune
    av_opt_set(codecCtx_->priv_data, "preset", "ultrafast", 0);

    // ---------- 关键：要求编码器生成全局头部 ----------
    codecCtx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // ---------- 重要：先打开编码器 ----------
    ret = avcodec_open2(codecCtx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Could not open codec, error: " << errbuf << std::endl;
        goto cleanup;
    }

    // ---------- 然后再将编码器参数复制到流 ----------
    ret = avcodec_parameters_from_context(videoStream_->codecpar, codecCtx_);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Failed to copy codec parameters to stream, error: " << errbuf << std::endl;
        goto cleanup;
    }

    // 7. 打开输出文件
    if (!(fmtCtx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmtCtx_->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
            std::cerr << "Could not open output file, error: " << errbuf << std::endl;
            goto cleanup;
        }
    }

    // 8. 写入文件头
    ret = avformat_write_header(fmtCtx_, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error writing header, error: " << errbuf << std::endl;
        goto cleanup;
    }

    // 9. 初始化像素转换器（OpenCV Mat BGR -> YUV420P）
    swsCtx_ = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    if (!swsCtx_) {
        std::cerr << "Could not initialize sws context" << std::endl;
        goto cleanup;
    }

    // 10. 分配 AVFrame 用于编码
    frame_ = av_frame_alloc();
    if (!frame_) {
        std::cerr << "Could not allocate frame" << std::endl;
        goto cleanup;
    }
    frame_->format = codecCtx_->pix_fmt;
    frame_->width  = codecCtx_->width;
    frame_->height = codecCtx_->height;
    ret = av_frame_get_buffer(frame_, 0);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Could not allocate frame buffer, error: " << errbuf << std::endl;
        goto cleanup;
    }

    return true;

cleanup:
    // 清理已分配的资源
    if (swsCtx_) sws_freeContext(swsCtx_), swsCtx_ = nullptr;
    if (frame_) av_frame_free(&frame_);
    if (codecCtx_) avcodec_free_context(&codecCtx_);
    if (fmtCtx_ && !(fmtCtx_->oformat->flags & AVFMT_NOFILE) && fmtCtx_->pb) {
        avio_closep(&fmtCtx_->pb);
    }
    if (fmtCtx_) avformat_free_context(fmtCtx_), fmtCtx_ = nullptr;
    videoStream_ = nullptr;
    return false;
}

/**
 * 写入一帧 cv::Mat（BGR 格式）
 * @param bgrMat OpenCV Mat，尺寸需与构造时一致
 * @return true 成功，false 失败
 */
bool MkvAllIntraWriter::writeFrame(const cv::Mat& bgrMat) {
    if (!codecCtx_ || !frame_ || !swsCtx_) {
        std::cerr << "Writer not opened or already closed" << std::endl;
        return false;
    }

    // 将 BGR Mat 数据转换为 YUV 并填充 frame_
    const int in_linesize[1] = { static_cast<int>(bgrMat.step) };
    sws_scale(swsCtx_, &bgrMat.data, in_linesize, 0, codecCtx_->height,
                frame_->data, frame_->linesize);

    // 设置 PTS（以帧计数 * time_base 计算）
    frame_->pts = frameCount_++;

    // 发送帧到编码器
    int ret = avcodec_send_frame(codecCtx_, frame_);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error sending frame: " << errbuf << std::endl;
        return false;
    }

    // 接收编码后的包
    AVPacket* pkt = av_packet_alloc();
    ret = avcodec_receive_packet(codecCtx_, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_packet_free(&pkt);
        return true; // 需要更多帧
    } else if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error receiving packet: " << errbuf << std::endl;
        av_packet_free(&pkt);
        return false;
    }

    // 设置包的时间基并写入文件
    pkt->stream_index = videoStream_->index;
    av_packet_rescale_ts(pkt, codecCtx_->time_base, videoStream_->time_base);

    ret = av_interleaved_write_frame(fmtCtx_, pkt);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error writing frame: " << errbuf << std::endl;
        av_packet_free(&pkt);
        return false;
    }

    av_packet_free(&pkt);
    return true;
}

/**
 * 关闭写入器，写入文件尾并释放资源
 */
void MkvAllIntraWriter::close() {
    if (codecCtx_ && videoStream_ && fmtCtx_) {
        // 刷新编码器（发送 NULL 帧以获取剩余包）
        avcodec_send_frame(codecCtx_, nullptr);
        AVPacket* pkt = av_packet_alloc();
        while (true) {
            int ret = avcodec_receive_packet(codecCtx_, pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;
            pkt->stream_index = videoStream_->index;
            av_packet_rescale_ts(pkt, codecCtx_->time_base, videoStream_->time_base);
            av_interleaved_write_frame(fmtCtx_, pkt);
        }
        av_packet_free(&pkt);

        // 写入文件尾
        av_write_trailer(fmtCtx_);
    }

    // 释放资源
    if (swsCtx_) {
        sws_freeContext(swsCtx_);
        swsCtx_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (codecCtx_) {
        avcodec_free_context(&codecCtx_);
    }
    if (fmtCtx_ && !(fmtCtx_->oformat->flags & AVFMT_NOFILE) && fmtCtx_->pb) {
        avio_closep(&fmtCtx_->pb);
    }
    if (fmtCtx_) {
        avformat_free_context(fmtCtx_);
        fmtCtx_ = nullptr;
    }
    videoStream_ = nullptr;
    frameCount_ = 0;
}

