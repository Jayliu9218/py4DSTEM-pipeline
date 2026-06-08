从截图看，当前 APP 已经把 notebook 流程拆成了几个功能页，基础框架是对的：**导入 → 虚拟成像 → Probe/Bragg disks → Calibration → Orientation → Strain**。但现在的问题是：它更像“把代码参数搬进 GUI”，还没有形成一个真正面向用户的 **4D-STEM 分析工作台**。建议从 **界面结构、流程逻辑、参数联动、高级功能扩展** 四个层面优化。

------

# 一、总体修改方向

当前 APP 的主要问题有 5 个：

1. **页面空白区域太多**，数据未加载时黑色图窗显得像程序异常；
2. **流程依赖不明显**，用户不知道必须先完成哪个步骤，才能做 orientation 或 strain；
3. **Calibration 页很关键，但现在像参数表，不像决策流程**；
4. **功能停留在基础分析**，缺少能体现 4D-STEM 特色的高级模块；
5. **日志和进度区没有发挥作用**，应成为“计算过程追踪 + 参数记录 + 错误提示”中心。

建议把 APP 从现在的：

```text
Notebook 参数界面
```

升级成：

```text
4D-STEM 数据分析 Pipeline + Scientific Workbench
```

------

# 二、主界面结构建议

现在上方 tab 是：

```text
1-3 Import, Load_Visualise
3.2 VBF / VADF
4-5 Probe_Bragg Disks
6 Calibration
7 Orientation Analysis
8 StrainMap
```

建议改成更清晰的一级导航：

```text
1 Data
2 Virtual Imaging
3 Probe & Bragg Disks
4 Calibration
5 Crystal Analysis
6 Strain Analysis
7 Advanced 4D-STEM
8 Export & Report
```

其中 **Advanced 4D-STEM** 可以包含：

```text
Structural Phase Mapping
DPC / COM
Ptychography
Amorphous / RDF / FEM
Pattern Clustering
```

这样用户会感觉这个软件不只是 EBSD-like orientation/strain，而是完整 4D-STEM 平台。

------

# 三、Import / Load / Visualise 页修改建议

## 当前问题

截图中左侧只有 HDF5 tree，右侧是 DataCube 信息和 attributes 表，中间上下两个黑色显示区。数据未加载时，界面显得空。

## 建议改法

### 1. 增加“数据角色指定”

HDF5 文件中往往有多个数据节点，例如：

```text
4DSTEM_AuNanoplatelet
4DSTEM_polyAu
defocused_CBED
vacuum_probe
```

建议在 tree 中右键或按钮设置角色：

```text
Set as Target DataCube
Set as Polycrystal Calibration Data
Set as Vacuum Probe
Set as Defocused CBED
Set as Reference Diffraction
```

右侧状态栏显示：

```text
Target DataCube: 4DSTEM_AuNanoplatelet
Poly Standard: 4DSTEM_polyAu
Vacuum Probe: vacuum_probe
Defocused CBED: defocused_CBED
```

这比让用户自己记变量名更适合 GUI。

------

### 2. 黑色预览区改成“有提示的空状态”

数据未加载时不要只显示黑图，建议显示浅灰提示：

```text
No DataCube loaded.
Open an HDF5 file and select a 4D-STEM dataset from the tree.
```

加载后，上图显示：

```text
Mean diffraction pattern / selected diffraction pattern
```

下图显示：

```text
Mean real-space image / virtual bright field preview
```

这样用户一眼知道上下图分别是什么。

------

### 3. 增加自动识别按钮

在 File 菜单旁边增加：

```text
Open HDF5
Auto Detect DataCube
Auto Detect Probe / Calibration Data
```

自动检测逻辑可以按 shape 判断：

```text
4D array → DataCube
2D array with diffraction-like shape → vacuum_probe / CBED / diffraction image
2D real-space shape → image
```

------

# 四、VBF / VADF 页修改建议

## 当前问题

现在 Mode 只有 Bright Field，参数是 center_x、center_y、inner_radius、outer_radius。功能太基础。

## 建议改成“虚拟探测器设计器”

左侧参数区分成 3 类：

### 1. Detector type

```text
Bright Field
Annular Bright Field
Annular Dark Field
High Angle ADF
Custom Annulus
Bragg Disk Dark Field
Segmented Detector
COM / DPC Preview
```

### 2. Detector source

```text
Use manual center
Use calibrated origin
Use measured origin map
```

如果用户已经做了 calibration，应默认使用 calibrated origin，而不是让用户手动填 `center_x=1.00`、`center_y=1.00`。

### 3. Detector preview

建议在 diffraction pattern 上叠加虚拟 detector 区域：

```text
圆形 BF mask
环形 ADF mask
选中的 Bragg disk mask
分区探测器 mask
```

现在用户看不到自己设置的 detector 在哪里，这会影响理解。

------

## 增加 q-resolved virtual image

这是 4D-STEM 比普通 STEM 更强的功能。建议增加：

```text
Generate q-series
```

例如：

```text
q = 0–5 px
q = 5–10 px
q = 10–20 px
q = 20–40 px
...
```

输出一组 virtual images，帮助判断不同散射角贡献。

------

# 五、Probe & Bragg Disks 页修改建议

## 当前问题

这个页面现在参数比较多，但缺少“检测质量评价”。用户只能 Run，却不知道参数好不好。

## 建议增加三个子区

### 1. Probe template 区

当前有按钮：

```text
Vacuum-Probe
Run Current Pattern
Run 6 Selected Pos
Run Full BraggVector
```

建议改成更清楚的流程按钮：

```text
1 Load / Generate Probe Template
2 Preview Detection on Current Position
3 Preview Detection on Selected Positions
4 Run Full Bragg Disk Detection
5 Generate Detection Quality Map
```

------

### 2. Bragg disk detection preview

当前右侧只是一张黑图。建议显示：

```text
当前 diffraction pattern
叠加检测到的 Bragg disk circles
中心束位置
edge boundary
minPeakSpacing circle
```

下面表格显示：

```text
index | qx | qy | intensity | distance from origin | selected
```

------

### 3. Detection quality maps

全图检测完成后，建议自动生成：

```text
Number of peaks map
Mean peak intensity map
Max peak intensity map
Detection failure mask
Bragg vector histogram
```

这些对于 orientation 和 strain 非常重要。

尤其是：

```text
peak count map
```

应该作为后续 orientation map 是否可信的第一张质量图。

------

# 六、Calibration 页修改建议

这个页面最需要改。现在它有 origin、ellipse、pixel、rotate 的状态，也有几个按钮，但用户还是很难判断“应该点哪个”。

## 建议改成“Calibration Decision Panel”

左侧上半部分保留状态，但增加颜色：

```text
Origin     Missing / Measured / Applied
Ellipse    Missing / Fitted / Transferred / Applied
Q pixel    Missing / Set / Fitted / Transferred / Applied
Rotation   Missing / Set / Applied
```

颜色建议：

```text
灰色：Not available
黄色：Measured but not applied
绿色：Applied
红色：Required but missing
```

------

## 增加分析目标选择

在 Calibration 页最上面加：

```text
Analysis target:
[ Preview / ACOM / Strain / Flowline / DPC / Ptychography ]
```

选择后自动提示需要哪些校正。

例如选择 ACOM：

```text
Required:
✓ Origin
✓ Q pixel size
Recommended:
△ Ellipse
Optional:
○ QR rotation
```

选择 Strain：

```text
Required:
✓ Origin
✓ Q pixel size
✓ Ellipse
✓ QR rotation
✓ Reference lattice
```

选择 DPC：

```text
Required:
✓ Origin / diffraction shift
✓ QR rotation
Recommended:
△ scan drift correction
```

这样用户不会盲目操作。

------

## 增加 calibration 来源选择

建议加入：

```text
Calibration source:
[ Read from file ]
[ Transfer from polycrystal standard ]
[ Fit from target dataset ]
[ Manual input ]
```

并且每种方式对应不同按钮。

比如：

### Transfer from polycrystal

```text
Use polycrystal for:
[✓] Q pixel size
[✓] Ellipse
[ ] Origin
[ ] QR rotation
```

默认不要 transfer origin 和 rotation，这和我们前面讨论一致。

------

## Calibration 页建议重排

当前按钮是：

```text
6.1 Measure/Fit Origin
6.2 Fit Ellipticity
6.3 Set Pixel Size
6.4 Set QR Rotation
Apply Selected Corrections
```

建议改成：

```text
1 Check Existing Calibration
2 Measure Origin from Target
3 Fit Ellipse from Poly / Target
4 Calibrate Q Pixel Size
5 Set / Compare QR Rotation
6 Apply Calibration
7 Validate Calibration
```

最后一定要有 **Validate Calibration**，自动输出：

```text
raw BVM vs calibrated BVM
ring circularity before / after
known q peak overlay
calibration summary table
```

这一步很关键。

------

# 七、Orientation Analysis 页修改建议

## 当前问题

现在只支持 Load CIF、Create Orientation Plan、Match and Show Orientation Map。功能是 ACOM 基础流程，但还不够完整。

## 建议增加 5 类结果图

Orientation map 不应该只输出一张彩色图。建议自动生成：

```text
Orientation RGB map
Correlation score map
Peak count map
Reliability / confidence map
Multi-match ambiguity map
```

尤其是：

```text
correlation score map
```

非常重要。否则用户不知道 orientation map 是不是可信。

------

## 增加 Phase Mapping

这是最值得加入的高级晶体功能。

当前页面可以改名为：

```text
Crystal Analysis
```

里面分为：

```text
ACOM Orientation
Structural Phase Mapping
Variant / Twin Analysis
```

### Phase Mapping 功能

允许加载多个 CIF：

```text
Phase 1: fcc Au
Phase 2: bcc Fe
Phase 3: oxide
Phase 4: hcp phase
```

每个 phase 建立 orientation plan，然后比较 correlation：

```text
phase label = phase with highest correlation
```

输出：

```text
phase map
orientation map per phase
phase confidence map
ambiguous phase mask
```

这比单一 orientation map 更体现 4D-STEM 高级价值。

------

## 增加 multi-match 诊断

当前有：

```text
matches to return = 1
minimum match angle = 5
```

建议加入解释性 UI：

```text
Return multiple matches for:
[ ] Overlapping grains
[ ] Twins / variants
[ ] Ambiguous indexing diagnosis
```

并自动显示：

```text
Match 1 orientation
Match 2 orientation
Match 3 orientation
Correlation gap map: corr1 - corr2
```

`corr1 - corr2` 比单纯看三张 orientation map 更有用。

------

# 八、StrainMap 页修改建议

## 当前问题

StrainMap 页现在参数较多，但对用户来说不够直观。尤其 reference ROI 很关键，但只能输入数字。

## 建议改成交互式 ROI 选择

不要只让用户填：

```text
reference ROI rx start
reference ROI rx end
reference ROI ry start
reference ROI ry end
```

应该允许在图上直接框选 reference region。

流程：

```text
1 Show virtual image / orientation map
2 Draw reference ROI
3 Extract reference lattice
4 Run strain map
```

------

## 增加 strain 质量控制

除了现在的：

```text
exx
eyy
exy
theta
```

建议增加：

```text
principal strain ε1
principal strain ε2
lattice rotation
fit residual map
valid mask
reference sensitivity check
```

其中最重要的是：

```text
fit residual map
```

它可以告诉用户哪些区域不符合单一晶格模型，可能是重叠晶粒、多相、非晶、缺陷核心或检测失败。

------

## 增加单位和色标控制

Strain map 应明确显示：

```text
strain unit: %
rotation unit: degree / mrad
color range: symmetric / percentile / manual
```

建议提供：

```text
Auto symmetric range around 0
Use percentile 1–99%
Manual min/max
```

否则 strain 图很容易被色标误导。

------

# 九、建议新增 Advanced 4D-STEM 页面

目前 APP 到 StrainMap 就结束了，显得还不够高级。建议新增一个 tab：

```text
9 Advanced 4D-STEM
```

里面先做成模块入口，不一定一次全部实现。

## 推荐模块顺序

### 1. Structural Phase Mapping

优先级最高，和你现在的 Bragg disk / ACOM 代码最接近。

功能：

```text
load multiple CIFs
build orientation plans
match each phase
compare correlation
output phase map
```

------

### 2. Pattern Classification

很实用，也比较容易实现。

输入：

```text
full diffraction patterns
或 Bragg vector features
或 radial profiles
```

算法：

```text
PCA
NMF
k-means
UMAP
Gaussian mixture
```

输出：

```text
diffraction pattern cluster map
cluster average diffraction patterns
feature importance / representative patterns
```

这个可以发现：

```text
相区
厚度区
倾斜区
缺陷区
边缘区
```

这比简单 orientation 更有新意。

------

### 3. DPC / COM

功能：

```text
COM_x
COM_y
DPC vector field
integrated phase
electric field proxy
```

前提提示：

```text
Requires origin / diffraction shift calibration.
Requires QR rotation for physical vector interpretation.
```

------

### 4. Polar / RDF / FEM

适合非晶或半晶数据：

```text
polar transform
amorphous strain
RDF
FEM variance
flowline texture
```

这类可以作为“Disordered Materials”模块。

------

### 5. Ptychography

先作为高级模块入口：

```text
Single Side Band Ptychography
Iterative Ptychography
Phase Reconstruction
```

同时加条件检查：

```text
probe overlap sufficient?
scan step size available?
probe known?
bright-field disk valid?
```

如果条件不满足，给出提示，而不是直接让用户运行。

------

# 十、底部 Activity Log / Calculation Process 修改建议

现在底部有：

```text
Activity Log
Calculation Process
```

但看起来只是空白日志。

建议左侧 Activity Log 记录：

```text
[time] Loaded file: xxx.h5
[time] Selected target datacube: shape=(100,84,125,125)
[time] Bragg disks detected: average peaks = 29.4
[time] Calibration applied: origin + pixel + ellipse
```

右侧 Calculation Process 应显示：

```text
Current step
Progress bar
Estimated time
CPU/GPU status
Parameters snapshot
Warnings
```

例如运行 ACOM 时：

```text
Step: Matching orientations
Progress: 43 / 100 scan rows
Current parameters:
    angle_step_zone_axis = 2°
    angle_step_in_plane = 2°
    corr_kernel_size = 0.08 Å^-1
Warnings:
    12.5% pixels have fewer than 3 peaks
```

这会非常有科研软件的感觉。

------

# 十一、全局交互建议

## 1. 每个页面增加 Save / Export

不要只在最后导出。每个模块都应该能导出：

```text
image: png / tif
data: npy / csv / h5
figure: svg / pdf
state: json
```

例如：

```text
Export virtual image
Export BraggVectors
Export calibration
Export orientation map
Export strain map
Export project report
```

------

## 2. 增加 Project State

建议引入一个项目状态文件：

```text
project.json
```

记录：

```text
data path
selected nodes
detection parameters
calibration parameters
orientation parameters
strain parameters
export paths
```

这样用户下次打开可以继续。

------

## 3. 增加 Basic / Advanced 模式

当前参数很多，新手容易困惑。

建议提供：

```text
Basic mode
Advanced mode
```

Basic mode 只显示核心参数：

```text
minRelativeIntensity
minPeakSpacing
maxNumPeaks
Q pixel size
ellipse q range
```

Advanced mode 再显示：

```text
template sigma
subpixel
corr kernel size
excitation error sigma
power intensity
coordinate rotation
```

------

# 十二、优先级修改清单

## 第一优先级：必须改

```text
1. 数据加载页增加角色指定：Target / Poly / Probe / CBED
2. Calibration 页增加状态颜色和分析目标提示
3. Bragg disk 页增加检测质量图：peak count map / BVM
4. Orientation 页增加 correlation map 和 reliability map
5. Strain 页增加交互式 reference ROI
6. 底部 Calculation Process 增加真实进度条和参数记录
```

------

## 第二优先级：增强科研价值

```text
1. Structural Phase Mapping
2. Bragg-disk-selected VDF
3. Pattern Classification
4. DPC / COM
5. Amorphous / RDF / FEM
```

------

## 第三优先级：高级拓展

```text
1. Ptychography
2. Multi-phase orientation relationship
3. Variant / twin analysis
4. Defect-sensitive residual mapping
5. Report auto-generation
```

------

# 十三、推荐最终 APP 功能结构

我建议最终改成下面这样：

```text
File
├── Open HDF5
├── Save Project
├── Load Project
├── Export Results
└── Generate Report

Tabs
├── 1 Data Manager
│   ├── HDF5 tree
│   ├── dataset role assignment
│   ├── metadata viewer
│   └── DP / real-space preview
│
├── 2 Virtual Imaging
│   ├── BF / ABF / ADF / HAADF
│   ├── custom detector
│   ├── Bragg selected VDF
│   └── q-resolved image series
│
├── 3 Probe & Bragg Disks
│   ├── probe template
│   ├── single-position detection
│   ├── selected-position test
│   ├── full detection
│   └── detection quality maps
│
├── 4 Calibration
│   ├── calibration status
│   ├── read / transfer / direct fit
│   ├── origin
│   ├── ellipse
│   ├── Q pixel size
│   ├── QR rotation
│   └── validation
│
├── 5 Crystal Analysis
│   ├── ACOM orientation
│   ├── structural phase mapping
│   ├── multi-match ambiguity
│   └── orientation confidence
│
├── 6 Strain Analysis
│   ├── reference ROI
│   ├── crystalline strain
│   ├── residual map
│   └── lattice rotation
│
├── 7 Advanced 4D-STEM
│   ├── DPC / COM
│   ├── Ptychography
│   ├── RDF
│   ├── FEM
│   ├── amorphous strain
│   └── pattern clustering
│
└── 8 Export & Report
    ├── export images
    ├── export arrays
    ├── export calibration
    ├── export BraggVectors
    ├── export orientation map
    └── generate analysis report
```

------

# 十四、总结建议

你现在的 APP 已经有了“能跑 py4DSTEM 基础流程”的雏形，但下一步重点不是继续堆按钮，而是：

```text
1. 把流程依赖做清楚
2. 把 calibration 状态做可视化
3. 把质量控制图加进去
4. 把高级 4D-STEM 模块作为单独页面扩展
5. 让用户知道每一步的结果是否可信
```

最关键的升级点是：

> 从“运行代码的 GUI”变成“带有数据状态、校正逻辑、质量评价和高级分析模块的 4D-STEM 科研工作流平台”。