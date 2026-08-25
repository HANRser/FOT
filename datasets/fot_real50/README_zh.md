# FoT 现实照片冻结测试集（real50）

## 1. 用途与论文依据

本目录用于 `Flow of Truth: Proactive Temporal Forensics for Image-to-Video Generation`
的独立现实照片测试。论文第 4.1 节说明：

- FoT 训练使用约 118K 张 MSCOCO 图像和 85K 个光流样本；
- 所有训练图像缩放到 512×512；
- 论文评测场景正是 Face、Camera、Animal、Human-Environment、Multi-Human；
- 论文第 4.4 节指出，Camera 的全局一致运动通常较容易捕获，而
  Human-Environment 与 Multi-Human 的非刚性、空间异质运动更难。

因此，本测试集五类各 10 张，既保持类别平衡，也在每类中覆盖不同主体尺度、
背景层次和潜在运动结构。它只用于冻结后的独立测试，不得并入训练集、验证集、
微调集、提示词调优集或人工挑选最佳 checkpoint 的开发集。

## 2. 目录结构

```text
fot_real50/
├── originals/          # 从来源 CDN 下载后未修改的源文件字节，五类各 10 张
├── test_512/           # 最终测试输入：512×512、8-bit RGB、PNG、嵌入 sRGB ICC
├── contact_sheets/     # 仅供人工复核，不应送入模型
├── selection.json      # 人工选图、场景、裁剪焦点参数
├── manifest.csv        # 便于表格查看的完整清单
├── manifest.json       # 便于程序读取的完整清单
└── duplicate_report.json
```

`originals/` 中的“源文件”指无裁剪、无缩放参数请求得到并原样落盘的照片文件，
并不代表相机 RAW。`test_512/` 由 EXIF 方向校正、ICC 到 sRGB 转换、焦点方形裁剪、
Lanczos 缩放和 PNG 保存得到；从不直接拉伸长宽比。

## 3. 类别构成

- `face`：单人正脸或轻微侧脸，男女各 5 张，含微笑、轻微转头和不同背景。
- `camera`：3 张城市街道、2 张建筑走廊、2 张山地景观、3 张室内空间。
- `animal`：3 只狗、3 只猫、2 只鸟、2 匹马；每张只有一个主要动物。
- `human_environment`：做饭 2、阅读 2、维修 2、运动 2、电脑 1、陶艺 1。
- `multi_human`：交谈/会议、握手、篮球、足球、家庭活动和棋局等互动。

## 4. 数据泄漏边界

所有 50 张照片都有独立的 Unsplash 落地页，来源域不是 Flickr。COCO 的已知
图像来源链是 Flickr，因此该选择策略显著降低与 COCO2017 train/val 的直接重合
风险；本目录内部还做了 SHA-256 和 dHash 近重复检查，结果为 0 对。

但项目仓库中没有队友实际轻量训练所使用的逐图文件清单或哈希，因此无法仅凭
论文和仓库对“私人训练副本是否含同图的裁剪、翻转、调色或压缩版本”作数学证明。
正式出结果前，队友必须将实际 train/val 文件清单或感知哈希与
`manifest.json` 对比；一旦命中，必须更换样本并重新冻结。该限制不能用人工记忆
代替。

建议在首次测试前记录 Git commit、模型 checkpoint 哈希、`manifest.json` 哈希，
然后禁止改动 `test_512/`。测试后也不得把失败样本回流到训练或验证集。

## 5. 来源与许可

照片均从各自 `page_url` 所列 Unsplash 照片页选取，许可统一记录为
[Unsplash License](https://unsplash.com/license)，整理日期为 2026-08-25。
该许可允许免费下载、复制、修改和使用照片，署名不是强制要求，但本数据集仍在
清单中保留摄影者名称和落地页。公开再分发前应再次检查落地页状态、隐私/肖像权
和所在机构的研究伦理要求；版权许可不自动等于模特肖像授权。

## 6. 质检字段

`manifest.csv/json` 逐图记录：

- 来源页、照片 URL、作者、许可与整理日期；
- 原图格式、模式、宽高、ICC 情况和 SHA-256；
- 实际方形裁剪框；
- 测试副本尺寸、模式、格式、位深、sRGB ICC、Alpha 状态和 SHA-256；
- 简单色彩/边缘统计，供批量发现黑白图或异常模糊；
- 数据来源链与冻结用途说明。

联系表已经逐张人工复核，确认无拼图、截图、海报、边框、明显水印、字幕或大面积
文字；主体清楚且曝光正常。自动边缘统计只用于发现异常，不能替代人工清晰度判断。

## 7. 复现处理

已有原图时重新生成测试副本：

```powershell
python tools/prepare_fot_benchmark.py --skip-download
```

重新下载缺失原图并生成全部产物：

```powershell
python tools/prepare_fot_benchmark.py
```

不要在模型测试代码中读取 `originals/` 或 `contact_sheets/`；唯一允许的模型输入目录
是 `test_512/`。
