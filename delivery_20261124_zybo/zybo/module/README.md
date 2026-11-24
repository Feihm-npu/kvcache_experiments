# Zybo 根模块（req_1120 的 Z4–Z8）

本目录是根模块的源码、板 1 上的实测输出和使用说明。**正式登记还没有做**：`lanes` 和 `t` 要等你们根据 Z2 数据给出；在收到之前，板 1 上没有用临时参数做过正式登记。下面的实板验证全部用"测试登记"完成：由 `module_enroll.py` 从本批已录下的同开机原始读数（`zybo/sameboot_pilot/`，回放后端）算出，用完即删，没有交付。

## 文件

| 文件 | 作用 |
|---|---|
| `module_core.py` | 两个固件映像共用的核心：位打包、BCH(t, m=13)、Toeplitz、HMAC、r/k_u 的存放、K 的校验、PUF 读取源、登记数据的校验 |
| `module_enroll.py` | **登记模式**固件映像（每块板只运行一次），按需求的 6 步实现 |
| `module_serve.py` | **部署模式**固件映像：只有 unlock / derive / lock 三个请求 |
| `module_client.py` | Z6 主机客户端：从 stdin 读一行请求，转发，原样打印回复，退出码 0 |
| `module_calibrate.py` | Z5：标定 c |
| `research_frames.py` | Z8 用的研究协议（PUF-COLP v1）读取命令 |
| `test_module.py` | 32 项测试；PC（Python 3.12 / numpy 2.5 / bchlib 2.1.3）和板子（Python 3.6.5 / numpy 1.13.3 / bchlib 2.1.3）上各通过一次 |
| `challenges_K.npy` | 冻结的 K，int8 (4096, 64)，原始字节 SHA-256 `6743daa6…acd5` |
| `frozen/` | 冻结的 bitstream 与研究固件各一份，与需求中的 SHA-256 一致 |
| `calibration.json` | 板 1 的 c 与实测 kdf_s |
| `validation/` | 实板验证的输出（见下） |
| `configs/` | J6 用的四个模块配置 |

部署时板上只需要 `module_core.py`、`module_serve.py`、`challenges_K.npy`、冻结的研究固件与 bitstream、该板的 `enrollment.json`。`module_enroll.py` 在登记后不应再出现在交付部署的板上。

## 链路（Z6）

以太网，TCP。部署服务监听板子的 3041 端口，一行 JSON 进、一行 JSON 出：

```
{"op": "unlock", "passphrase_hex": ".."} -> {"success": true, "device_id": "..", "timing": {"acquire_s": .., "fe_s": .., "kdf_s": ..}}
                                             或 {"success": false, "device_id": "..", "error": "reconstruction"}
{"op": "derive", "label_hex": ".."}      -> {"success": true, "device_id": "..", "key_hex": "<64 位十六进制>"}
                                             或 {"success": false, "device_id": "..", "error": "locked"}
{"op": "lock"}                           -> {"success": true, "device_id": ".."}
其他任何输入                               -> {"success": false, "device_id": "..", "error": "unsupported_request"}
```

`module_client.py` 与 `jetson_package` 里 `CommandRootModule` 的调用方式一致：

```json
{"backend": "hardware", "device_id": "210251A08870",
 "command": ["python3", "module_client.py", "--host", "192.168.1.2", "--port", "3041"], "timeout_seconds": 60}
```

客户端把 stdin 的**一整段**内容当作一行请求（多行或非 JSON 则拒绝），连接板子，读到一行回复后原样写到 stdout。连不上或回复不是一行 JSON 时 stdout 为空、退出码 1。口令只经过 stdin、socket 和网线，不进 argv、文件或日志。`configs/` 里有 J6 四个配置的写法：`owner.json` 可直接用于板 1；`other`、`moved` 需要板 2 的地址和登记；`modified` 在板 1 上再起一个服务实例（另一个端口、另一页 OCM，见下）。

## 约定与两个需要你们确认的点

- **lane 编号**：`lanes` 是 0–7 的升序列表，0–3 为 `apuf_lanes` 的 0–3，4–7 为 `ff_lanes` 的 0–3；响应是指定 lane 的逐位 XOR。跨两族的 `lanes`（例如 `[0, 4]`）一次 unlock 读两遍 K，各族一遍。
- **t 的上限是 64**：`bchlib.BCH(t, m=13)`（板上是 bchlib 2.1.3）对 t > 64 直接报 `unable to inititalize bch, invalid parameters`。本批同开机数据里（`validation/reconstruction_matrix.json`，只是固件验证，不是你们的分析）：单条 lane 相对登记字的差错约 1–26 位，两条 14–38 位，apuf 四条 XOR 17–46 位，ff 四条 XOR 36–73 位（t = 64 时 91 次里失败 2 次），八条全 XOR 58–114 位（t = 64 时 91 次里失败 89 次）。真断电和整体加热后的差错只会更大，所以 `lanes` 选得宽时 t = 64 可能不够。
- `device_id` 用板序列号；J6 的 `moved` 配置里 `device_id` 要写**登记所属的板**（板 1）的序列号，因为服务回复的是登记文件里的 `device_id`。

## 部署模式的约束及其依据

| 约束 | 实现 | 依据 |
|---|---|---|
| 只接受三种请求 | `module_serve.py` 的 `handle_line` 逐项校验 op 和字段集合；其余一律回 `unsupported_request` | 接口本身的限制（代码强制）。实板测试：停掉研究服务后，100 条研究协议读取命令和 10 条 JSON 伪装全部被拒，服务日志里 `evaluations` 合计为 1，即随后的那一次 unlock（`../interface/closure_210251A08870.json`） |
| 研究协议帧 | 一收到 magic `PUF1` 立即拒绝并断开，不等换行 | 同上 |
| 不评估 K 以外的挑战 | 板上没有任何传入挑战的命令；每次评估都是对板上 K 的一遍读取 | 同上；`evaluations` 计数只在 unlock 时增加 |
| K 固化并核对 | K 从 `--k` 文件读取，监听之前和每次评估之前各核对一次原始字节 SHA-256，不符即拒绝 | 代码强制（`module_core.check_k`、`MmioSource._verify_k`） |
| 纠错失败与标签不符同一错误 | 两处失败走同一条返回路径，本地日志也只记 `unlock_failed` | 代码强制；测试 `test_failed_unlock_logs_no_distinction` |
| 登记与部署分属两个映像 | 两个入口文件；登记逻辑不在部署映像里 | 流程约定 + 文件划分；板上仍有 root SSH、Jupyter 和 JTAG（PYNQ 镜像），**有管理权限的人可以绕过这一切**，这是 PS 软件限制，不是硬件隔离 |
| r、k_u、w、w' 不落盘、不进日志 | 文件里只写 `enrollment.json`（公开数据）；日志只有事件名、计时和计数 | 代码强制；测试对日志做了密钥、根、口令十六进制的扫描 |
| r、k_u 只放在片上存储 | 常驻状态放在 OCM 的一页（默认 `0xFFFE8000`，一实例一页，`--ocm-address` 指定）：unlock 时算出的 r 先写入 OCM 再清掉进程里的副本，k_u 在 unlock 与 lock 之间只存在于 OCM；w、w' 是进程内的 numpy 数组，用后覆写为 0。每次取用都会在进程内存里短暂出现一份可覆写的副本（`bytearray`，用完清零）；Python 的 hashlib/OpenSSL 内部临时状态和小对象分配器不保证擦除，代码里尽力复用并清零（`scrub`），**不能视为 guaranteed**。服务退出（含 SIGTERM）时整页清零；被 SIGKILL 时残留会在下次启动时被识别并擦除 | 实板：`/proc/iomem` 显示该页只属于 `f800c000.ocmc`；页内在本批之前全零；两页在服务停止后读回全零（`validation/ocm_check.json`）。板上不可用时会拒绝启动，需显式改用 `--secret-store memory`（进程内存），此时必须在说明里写明 |

## 实板上的验证（板 1，2026-11-24，同一次开机）

| 输出 | 内容 |
|---|---|
| `calibration.json` | **Z5**：c = 4049，10 次 `kdf_s` 最小 1.083 s、中位数 1.085 s；每次迭代约 267 µs。只在板 1 上标定，两块板要用同一个 c（板 2 到位后重标，取较大者） |
| `validation/live_acquire.json` | 登记映像的真实读取路径（`--dry-run`，经冻结研究固件的 `measure_apuf_lanes` / `measure_ff_lanes` 读真 PL）：一遍 K × 两族共 18 次读取，每遍（两族）约 1.06 s；没有任何输出或文件 |
| `validation/reconstruction_matrix.json` | 16 组测试登记 (lanes, t)，每组 91 次 unlock 尝试（11 次空闲读数、80 次同开机升温扫描的读数，57.0–66.4 °C）。全部满足"成功当且仅当相对登记字的差错数 ≤ t" |
| `../timing/unlock_timing_210251A08870_replay_test.json` | **Z7（临时）**：连续 20 次 unlock 后再 1 次 unlock + 20 次 derive + lock，每次请求都经 `module_client.py` 从 PC 发出。`fe_s` 56–66 ms，`kdf_s` 1.09–1.15 s，unlock 往返 1.17–1.24 s，derive 往返中位数 23 ms（含在 PC 上启动客户端进程的时间）。`acquire_s` 是回放时间，不是读 PL 的时间 |
| `../interface/closure_210251A08870.json` | **Z8（临时）**：见上表 |
| `validation/j6_two_services.json` | 板上同时运行两个部署服务（3041 和 3043，两页 OCM）：正常的那个 unlock 成功；用 `ecc_hex` 改了 1 个字节的登记启动的那个回 `reconstruction`。即 J6 的 `owner` 与 `modified` 两种设置的布局 |

这些结果证明固件在这块板的环境里按需求运行；Z7、Z8 的正式测量要等正式登记，并在上电之后重做。

## 收到 lanes 和 t 之后

板子上（root；`bchlib` 装在 xilinx 用户的 site-packages 里，所以 root 要带 `PYTHONPATH`）：

```bash
export P="sudo -n env PYTHONPATH=/home/xilinx/.local/lib/python3.6/site-packages PYTHONDONTWRITEBYTECODE=1 python3"
M=/home/xilinx/puf/req1120_20261124/module      # 本目录在板上的位置；也可以自行上传

# 每次上电后：部署服务（默认先用冻结 bitstream 配置 PL；K 启动时校验）
$P $M/module_serve.py --enrollment <该板的 enrollment.json> --k $M/challenges_K.npy \
    --research $M/frozen/puf_server.py --bit $M/frozen/system_wrapper_apuf_ff_20260917.bit \
    --iterations <c> --port 3041

# 登记（每块板一次；输出文件不得已存在；只打印一行不含 w、r 的摘要）
$P $M/module_enroll.py --lanes 0,1,2,3 --t 64 --device-id <板序列号> --k $M/challenges_K.npy \
    --research $M/frozen/puf_server.py --bit $M/frozen/system_wrapper_apuf_ff_20260917.bit \
    --out <板序列号>_enrollment.json
```

登记和部署都要独占 PL，运行前停掉研究服务（port 3040）。`--backend replay`、`--backend sim` 和 `--secret-store memory` 只用于测试，部署时不要用。
