# Roban 人体跟随 · 阶段一

本包在 Ubuntu 20.04 / ROS Noetic 的 Roban NUC 上处理 D435 RGB-D 数据。YOLOX-Nano 检测人体，ByteTrack 关联轨迹，躯干区域的有效深度中位数给出距离。`observe` 与 `live` 共用处理链，两种模式均只输出决策、监控图像和 CSV。默认使用 `observe`。

## 在 Roban 上部署

在机器人本机执行以下命令。若 `src/roban_person_follower` 已是本项目的检出，使用 `git -C src/roban_person_follower pull --ff-only` 更新。

```bash
source /opt/ros/noetic/setup.bash
cd ~/robot_ros_application/catkin_ws
git clone https://github.com/Wuwhile/roban_person_follower.git src/roban_person_follower
python3 -c 'import cv2, numpy, scipy, onnxruntime; print(onnxruntime.get_available_providers())'
catkin_make --pkg roban_person_follower
source devel/setup.bash
rosrun roban_person_follower download_model.py
```

依赖 ROS 的 `rospy`、`message_filters`、`nodelet`、`image_proc`、`depth_image_proc`、`sensor_msgs` 和 `std_msgs`，以及 Python 的 NumPy、SciPy、OpenCV 和 ONNX Runtime。现场环境记录的 ONNX Runtime 版本为 1.16.3。权重下载工具使用 [Megvii 的 YOLOX-Nano 发布文件](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx) 并校验 SHA-256。若真机无法访问 GitHub，可从有网络的机器下载后复制到真机，再通过 `model_path:=/absolute/path/yolox_nano.onnx` 指定权重。

## 启动与监控

使用机器人现有的 ROS Master 和 D435 驱动。先确认两路图像、两路 `CameraInfo` 正常发布，彩色与深度光学坐标系间的 TF 可用。默认话题和阈值见 `launch/follower.launch` 及 `config/follower.yaml`。

```bash
roslaunch roban_person_follower follower.launch
```

在另一个已载入相同 Catkin 环境的机器人终端中执行命令。启动后默认待机，收到 `start` 才开始决策。`stop` 保留选中的目标 ID 并回到待命，`end` 释放目标。

```bash
rostopic echo /person_follower/status
rostopic pub -1 /person_follower/command std_msgs/String "data: 'start'"
rostopic pub -1 /person_follower/command std_msgs/String "data: 'stop'"
rostopic pub -1 /person_follower/command std_msgs/String "data: 'end'"
```

`/person_follower/target` 给出当前目标框、轨迹及深度有效性。`/person_follower/status` 给出启动模式、决策、停止原因和相机到决策发布的延迟。`/person_follower/debug_image` 在校正后的彩色图中标注所有当前确认轨迹并高亮选中目标。使用真机现有的 `web_video_server`，将 `ROBAN_HOST` 替换为可 SSH 登录的机器人地址，并在电脑上执行以下命令。8080 是服务默认端口，若真机改过端口，使用实际值。

```bash
ssh -NT -L 127.0.0.1:18080:127.0.0.1:8080 lemon@ROBAN_HOST
```

浏览器打开 `http://127.0.0.1:18080/stream_viewer?topic=/person_follower/debug_image`。

## 现场验收

1. 在 `observe` 里发送 `start`。让单名参与者保持在相机视野内，从正面走到左右两侧后回中。应依次看到对应转向与保持。
2. 让参与者在画面中央先远离至 1.7 m 以上，再靠近到 1.3 m 以内。画面应显示 `forward`、`hold` 和 `stop / too_close`。让参与者短暂离开并返回，核对 `target_lost` 和恢复后的决策。
3. 在近 1.5 m 处让参与者居中站定，沿相机前向用卷尺量相机到胸前的距离。在一段稳定的 CSV 记录中取 `distance_m` 中位数，记下时间段、卷尺读数、测距中位数和绝对误差。
4. 用 `stop` 结束决策。打开启动时打印的日志路径，默认位于 `~/.ros/roban_person_follower/logs/`。运行 `rosrun roban_person_follower summarize_run.py <本次CSV>` 得到有效决策帧率、延迟中位数与 P95。CSV 的 `frame` 行记录处理后的决策，`command` 和 `watchdog` 行记录独立停机事件。延迟止于 ROS 决策发布。
5. 结束 `observe` launch 后用 `mode:=live` 再启动。检查 `rostopic info /person_follower/status` 所示节点的发布项，只应出现本包的结果话题和 ROS 日志。两个模式使用相同的启动和停止命令。

若程序进入 `frame_timeout`，先分别核对 `/person_follower/rectified_color` 和 `/person_follower/depth_registered/image_rect` 的频率、帧时间戳及尺寸，再检查两路 `CameraInfo` 和 TF。对齐节点保留原深度时间戳，配对容差初值为 25 ms，帧超时为 0.5 s。

## 本地验证

在源码根目录执行。普通单元测试使用合成输入。从环境变量提供官方权重和一张含人体的照片后，同一套测试会额外运行真实 CPU 推理。`ROBAN_TEST_MODEL` 和 `ROBAN_TEST_IMAGE` 均可使用绝对路径。运行本地测试需要 NumPy、SciPy、OpenCV 与 ONNX Runtime。

```bash
export PYTHONPATH="$PWD/src"
python3 scripts/download_model.py --output logs/yolox_nano.onnx
curl -fLsS https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg -o logs/bus.jpg
export ROBAN_TEST_MODEL="$PWD/logs/yolox_nano.onnx"
export ROBAN_TEST_IMAGE="$PWD/logs/bus.jpg"
python3 -m unittest discover -s tests -v
```

ByteTrack 使用 [上游提交 d1bf019](https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99) 的 Python 跟踪器。输入适配为 NumPy，以 NumPy 和 SciPy 实现两轮 IoU 关联，沿用上游的卡尔曼滤波与轨迹管理。上游许可证见 `src/roban_person_follower/bytetrack/LICENSE`。
