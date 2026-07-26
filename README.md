# AutoDance Lab

AutoDance Lab 是一个本地运行的舞蹈队形分析工具。本次交付仅包含 **Phase 1：视频人物检测和追踪**，不分析动作、姿态类别、手势或舞步。

## 当前功能

- 上传固定机位的 MP4 舞蹈练习视频；
- 默认使用 `yolo11s-pose.pt` 检测人物和人体关键点；
- 使用开启 ReID 的舞蹈专用 BoT-SORT 分配并维持人物 ID；
- 允许低置信度检测续接已有轨迹，同时限制其创建新 ID；
- 根据固定人数、衣着外观和运动连续性离线归并碎片 ID；
- 用脚踝中点表示人物二维像素位置，脚踝不可用时退化为检测框底边中心；
- 生成带人物框、位置点和 ID 的 MP4 预览；
- 输出逐帧人物位置 `tracks.json`；
- 将异常写入 `analysis.log`，单帧失败时继续处理后续画面。

网格映射、队形变化检测、SVG 编辑和 PDF/SVG 导出属于后续 Phase，当前版本尚未实现。

## 环境要求

- Python 3.10 或 3.11（推荐）；
- 建议使用虚拟环境；
- 首次运行分析时需要联网，Ultralytics 会自动下载 `yolo11n-pose.pt`；
- CPU 可以运行，但长视频会较慢；安装匹配本机 CUDA 的 PyTorch 后可自动使用 GPU。

## 安装

进入项目目录：

```bash
cd autodance
python -m venv .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

浏览器访问终端显示的地址，通常为 `http://127.0.0.1:7860`。程序监听 `0.0.0.0`，也适用于 Hugging Face Spaces 和 ModelScope 容器环境。

## 输出文件

每次分析会创建独立目录：

```text
data/runs/<时间_任务ID>/
├── tracked_preview.mp4
├── tracks.json
└── analysis.log
```

`tracks.json` 是逐帧对象数组，每一项格式如下：

```json
{
  "frame_id": 0,
  "timestamp": 0.0,
  "persons": [
    {
      "id": 1,
      "x": 123,
      "y": 456
    }
  ]
}
```

这里的 `id` 是离线归并后的最终人物编号，范围通常为 `1..固定人数`；
`x`、`y` 是当前 Phase 的视频像素坐标。人物暂时被遮挡或检测置信度
不足时，该帧中仍可能暂时缺少对应人物。

## 使用建议

- 使用固定机位、全身可见、光线充足的视频；
- 尽量避免人物长时间完全互相遮挡；
- 先用较短视频验证效果；
- 默认最低检测置信度为 `0.10`，低置信度框只用于续接轨迹；不要因为 ID
  过多而盲目提高此值，否则会增加轨迹断裂；
- 固定机位多人舞蹈建议使用默认的 `Dance BoT-SORT`；
- “视频中的固定人数”必须填写准确，程序会据此把在线碎片轨迹归并为最终身份；
- 更换追踪器会从视频开头重新分析，不会沿用之前的 ID；
- `yolo11s-pose` 比 nano 模型更准确但更慢；有独立显卡时可进一步改为
  `yolo11m-pose.pt`。

## 项目结构

```text
autodance/                 # 项目名称：AutoDance Lab
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── model_loader.py
├── trackers/
│   └── dance_botsort.yaml  # 固定机位舞蹈专用追踪参数
├── vision/
│   ├── detector.py
│   ├── identity.py         # 固定人数离线身份归并
│   ├── tracker.py
│   └── coordinate.py       # Phase 2 占位
├── formation/
│   ├── analyzer.py         # Phase 3 占位
│   └── grid.py             # Phase 2 占位
├── visualization/
│   ├── svg_generator.py    # Phase 4 占位
│   └── pdf_export.py       # 后续 Phase 占位
└── data/               # 中间结果与分析输出
```
