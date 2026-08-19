# AutoDance Lab

AutoDance Lab 是一个本地运行的舞蹈队形分析工具。当前包含 **Phase 1：
视频人物检测和追踪**、**Phase 2：透视标定和 9×9 舞台网格映射**，
以及 **Phase 3：关键队形检测**。暂不分析动作、姿态类别、手势或舞步。

## 当前功能

- 上传固定机位的 MP4 舞蹈练习视频；
- 默认使用 `yolo11s-pose.pt` 检测人物和人体关键点；
- 使用开启 ReID 的舞蹈专用 BoT-SORT 分配并维持人物 ID；
- 允许低置信度检测续接已有轨迹，同时限制其创建新 ID；
- 根据固定人数逐帧执行一对一身份分配，以衣着外观为主、运动连续性为辅，
  修复在线轨迹断裂和交叉换 ID；
- 在视频首帧手动选择左上、右上、右下、左下四个舞台角点；
- 将脚点从视频像素坐标透视转换为连续舞台坐标和 9×9 网格行列；
- 将短时漏检、越界脚点和大幅单帧漂移视为缺测，用前后有效位置插值修复；
- 用脚踝中点表示人物二维像素位置，脚踝不可用时退化为检测框底边中心；
- 生成带透视网格、人物框、位置点、ID 和格位的 MP4 预览；
- 输出最终位置 `tracks.json`、诊断用 `raw_tracks.json` 和可复用的
  `calibration.json`；
- 在稳定时间窗口中提取关键队形，过滤单人移动、短时抖动、漏检和过多插值；
- 支持将关键队形离散到 8–40 范围内的自定义网格，默认使用 20×20；
- 输出关键队形时间点、确认时间、人物 ID 和格位到 `formations.json`；
- 将异常写入 `analysis.log`，单帧失败时继续处理后续画面。

SVG 队形图、队形编辑和 PDF/SVG 导出属于后续 Phase，当前版本尚未实现。

## 环境要求

- Python 3.10 或 3.11（推荐）；
- 建议使用虚拟环境；
- 首次运行分析时需要联网，Ultralytics 会自动下载 `yolo11s-pose.pt`；
- CPU 可以运行，但长视频会较慢；安装匹配本机 CUDA 的 PyTorch 后可自动使用 GPU。

## 安装

进入项目目录：

```bash
cd autodance
.\autodance_env\Scripts\Activate.ps1
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
├── raw_tracks.json
├── formations.json
├── calibration.json
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
      "y": 456,
      "stage_x": 3.125,
      "stage_y": 6.42,
      "grid_col": 4,
      "grid_row": 7,
      "in_stage": true
    }
  ]
}
```

这里的 `id` 是离线归并后的最终人物编号，范围通常为 `1..固定人数`；
`x`、`y` 是视频像素坐标；`stage_x`、`stage_y` 是以一个网格单元为
单位的连续舞台坐标；`grid_col`、`grid_row` 是从 1 开始的离散格位。
人物在标定舞台之外时，`in_stage` 为 `false`，行列为 `null`。
短时异常被修复时会额外包含 `"interpolated": true`；预览中的格位标签会
显示 `*`，便于区分模型直接观测和插值位置。

`calibration.json` 保存四个像素角点、网格定义和像素到舞台的单应性矩阵，
可以用于核查或在后续处理程序中复现相同坐标变换。

`raw_tracks.json` 保存未经最终身份分配的在线 ID、检测框、置信度和脚点，
用于排查漏检、重复框和在线换 ID，不应直接用于队形变化分析。

## 关键队形检测

Phase 3 从稳定后的 `tracks.json` 提取关键队形，不直接使用在线追踪结果。
默认判定规则如下：

- 所有人在连续 `0.75` 秒内至少出现 80%；
- 单个人的插值位置占比不超过 30%；
- 每个人相对窗口中位位置的 80 分位偏差不超过 1.25 个目标网格单位；
- 至少两个人相对上一队形移动 2 个目标网格单位，才进入队形转换；
- 转换后重新稳定满一个窗口，才确认新队形；
- 单人动作、短暂越界、单帧漂移和未稳定的持续运动不会生成新队形。

界面中的“队形网格宽度/高度”只控制 `formations.json` 的离散坐标范围，
可选 8–40，默认 20×20；它不会改变 Phase 2 的 9×9 透视标定网格。

`formations.json` 是按时间排序的关键队形数组：

```json
[
  {
    "formation_id": 1,
    "time": 0.375,
    "confirmed_at": 0.75,
    "frame_id": 9,
    "grid_width": 20,
    "grid_height": 20,
    "persons": [
      {"id": 1, "x": 15, "y": 12},
      {"id": 2, "x": 5, "y": 12}
    ]
  }
]
```

`time` 和 `frame_id` 指向稳定窗口的中间位置，适合展示该队形；
`confirmed_at` 是系统积累足够稳定帧并确认队形的时间。人物 `x`、`y` 是
从 1 开始、且受 `grid_width` / `grid_height` 约束的队形图坐标，不是视频
像素坐标，也不是 Phase 2 的连续 `stage_x` / `stage_y`。

进入 Phase 4 前，应在最新运行中确认 `formations.json` 队形数量合理，并在
`tracked_preview.mp4` 的各个 `time` 时间点目视核对队形和人物 ID。Phase 4
将以这份已验证的数据生成 SVG，而不会重新检测队形。

## 透视标定

1. 上传视频，等待程序显示首帧；
2. 严格按照左上、右上、右下、左下的顺序点击舞台区域四角；
3. 画面出现完整 9×9 网格即表示标定有效；
4. 如点击错误，使用“重新标定”清空四点；
5. 填写视频固定人数并开始分析。

角点应该选在舞者脚部实际活动的地面平面上，而不是背景墙、画面四角或
具有高度的物体上，否则透视后的格位会产生系统偏差。

## 使用建议

- 使用固定机位、全身可见、光线充足的视频；
- 尽量避免人物长时间完全互相遮挡；
- 先用较短视频验证效果；
- 默认最低检测置信度为 `0.10`，低置信度框只用于续接轨迹；不要因为 ID
  过多而盲目提高此值，否则会增加轨迹断裂；
- 默认以 `imgsz=960`、NMS IoU `0.50` 推理；测试中比原来的 640 输入显著
  提高拥挤场景下的四人完整率，但 CPU 处理时间约增加一倍；
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
│   └── coordinate.py       # 四点透视坐标转换
├── formation/
│   ├── analyzer.py         # 关键队形稳定窗口与转换检测
│   └── grid.py             # 9×9 网格定位和视频叠加
├── visualization/
│   ├── svg_generator.py    # Phase 4 占位
│   └── pdf_export.py       # 后续 Phase 占位
└── data/               # 中间结果与分析输出
```
