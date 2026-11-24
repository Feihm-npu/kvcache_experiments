# 硬件实验需求（2026-11-20，11-24 合并）：Jetson、Zybo 与 U200

本文分三部分，各平台可以分头进行。硬件方只交原始数据和程序输出，分析由我方完成。

| 部分 | 平台 | 内容 | 依赖 |
|---|---|---|---|
| J | Jetson Orin | 保护后解码的时延、GPU 时间、kernel 时间、能耗、冷层搬移；绑定测试 | J6 需要 Z 部分的根模块 |
| Z | 两块 Zybo Z7-10 | 断电与温度采集、跨板 CRP、板上根模块、解锁耗时、接口关闭 | Z4–Z8 需要我方根据 Z2 数据给出的登记参数 |
| U | Alveo U200 + DRAM Bender | DDR4 降时序失效图样的跨芯片、跨断电采集 | 无 |

建议顺序：**Z2 最先做**（我方据此给出登记参数），J1–J5、U0–U1、Z3 可同时进行；Z4–Z8 完成后做 J6。

---

## J. Jetson Orin

**代码**：本仓库 `jetson_package/`。不要修改 kernel 和脚本；任何检查失败都把完整日志交回，不要绕过。

**模型**：`Qwen/Qwen3-0.6B`，提前下载到本地 HF cache，下文 `$M` 指 `models--Qwen--Qwen3-0.6B` 目录。所有命令都显式给出 `--out`。

**环境**：

```bash
cd jetson_package
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch 用 NVIDIA 为对应 JetPack 提供的 wheel，见文件内注释
export PYTHONPATH=$PWD/src
sudo nvpmodel -m 0 && sudo jetson_clocks   # 最高性能模式
```

| 编号 | 内容 | 命令 | 产物 |
|---|---|---|---|
| J0 | **平台与门控**（必须先做）。记录模块型号、JetPack 版本、功耗模式和内存；Triton 冒烟测试和单元测试必须通过，否则停下并交回输出 | `cat /proc/device-tree/model`；`cat /etc/nv_tegra_release`；`nvpmodel -q`；`jetson_clocks --show`；`free -b`；`pip freeze`；`python scripts/triton_smoke.py`；`python -m pytest -q tests` | 各命令输出、`triton_smoke.json`、pytest 输出 |
| J1 | **端到端解码时延**：plain、quant、ring 三种配置交错执行，batch 1 | `python scripts/run_ring_e2e_verified.py --model-cache-dir $M --tag qwen3_orin --prefills 1024 2048 4096 --repeats 5 --out ring_e2e_verified_qwen3_orin.json` | 输出 JSON |
| J2 | **GPU 时间** | `python scripts/profile_ring_decode.py --model-cache-dir $M --tag qwen3_orin --prefills 1024 2048 4096 --out ring_decode_profile_qwen3_orin.json` | 输出 JSON |
| J3 | **注意力 kernel 时间**：Qwen3-0.6B 全部层的几何（默认参数） | `python scripts/run_kernel_baselines.py --tag orin_qwen3step --contexts 1024 2048 4096 --out kernel_baselines_orin_qwen3step.json` | 输出 JSON |
| J4 | **每 token 能耗**：只对解码步计量；空闲基线单独记录，不要自行扣除 | `python scripts/jetson_power.py --probe`；`python scripts/run_jetson_energy.py --model-cache-dir $M --tag qwen3_orin --prefills 1024 2048 4096 --out jetson_energy_qwen3_orin.json` | probe 输出、输出 JSON |
| J5 | **冷层搬移**：一个 CPU 核上的封装、打开、热层载入，以及与重算 prefill 的比较；`--e2e` 用 J1 的输出 | `python scripts/run_cold_tier_cost.py --model-cache-dir $M --tag qwen3_orin --prefills 1024 2048 4096 --dtypes fp32 --threads 1 --e2e ring_e2e_verified_qwen3_orin.json --out cold_tier_cost_qwen3_orin.json` | 输出 JSON |
| J6 | **绑定测试**（Z4–Z6 完成后）：Jetson 连接两块 Zybo 的根模块。口令自定，写在一个文件里，不要交付。需要四个模块配置（格式见 Z6）：`owner` 为板 1；`other` 为板 2 及其自己的登记；`moved` 为板 2 启动时加载**板 1 的** `enrollment.json`；`modified` 为板 1 启动时加载一份 `ecc_hex` 中改动了 1 个字节的登记 | `python scripts/run_root_module_binding.py --owner owner.json --other other.json --moved moved.json --modified modified.json --passphrase-file <口令文件> --trials 20 --context 2048 --model-cache-dir $M --out root_module_binding_qwen3_orin.json` | 输出 JSON、四个模块配置 |

`jetson_package/configs/` 里有模块配置的模板；用模拟器配置可以在接上板子之前先跑通 J6 的流程。

---

## Z. Zybo Z7-10

### 冻结的输入

| 项 | 冻结值 |
|---|---|
| bitstream | `system_wrapper_apuf_ff_20260917.bit`，SHA-256 `224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd` |
| 研究模式固件 | `puf_server.py`，SHA-256 `a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5` |
| 电路 | 4 条 arbiter lane 和 4 条 feed-forward arbiter lane，按 lane 读取（family `apuf_lanes`、`ff_lanes`） |
| 挑战集 K | `challenges.npy`（SHA-256 `69324bbc2daf94b7b9c401f3e912f2a675b59bdac9adcd87e0abff3a5ddd1905`）的前 4096 行，shape (4096, 64)，int8；`np.ascontiguousarray(c[:4096]).tobytes()` 的 SHA-256 为 `6743daa679eaf86c3cba25c028728774bcf4a80f214b17d9b9e2e31cec55acd5` |

- 两块板都只烧这个 bitstream，不重新综合、不重新布局、不修改 PL。
- 研究模式的读取路径保持不变；固件若有修改，记录修改后的 SHA-256。
- 每次上电都记录实际加载的 bitstream 和固件的 SHA-256。

### 采集（P0）

| 编号 | 内容 |
|---|---|
| Z1 | **板子标识**：两块同型号板（XC7Z010-1CLG400）的序列号、MAC，以及（若可读）Device DNA。板 1 为此前采集所用的板（eth0 MAC `00:0a:35:00:1e:53`），请核对 |
| Z2 | **断电与温度（最先做）**：每块板在室温和约 60 °C 下各做 20 次**真断电**，即切断电源且断电不少于 60 s；复位、软重启、重新加载 bitstream 都不算。每次上电后用研究模式对 K 读 1 遍，按 lane 输出 `apuf_lanes` 和 `ff_lanes`。每次上电记录：断电时长、上电到读取开始的秒数、读取时的 XADC 温度、`boot_id`、bitstream 与固件的 SHA-256 |
| Z3 | **跨板 CRP**：板 2 用研究模式采集 100,000 个挑战，由 `np.random.default_rng(20261120).integers(0, 2, (100000, 64), dtype=np.int8)` 生成，去掉与 K 重复的行；每个挑战读 1 遍，按 lane 输出 `apuf_lanes` 和 `ff_lanes` |

### 根模块（P0，收到我方登记参数后）

登记参数 `lanes`（参与 XOR 的 lane）和 `t`（BCH 纠错预算）由我方根据 Z2 数据给出。

**Z4 登记与部署固件**：在板上新增一个**模块服务**，与研究模式分开运行。

**登记模式**（每块板只运行一次）：

1. 对 K 读 9 遍，按 `lanes` 把指定 lane 的输出逐位 XOR，得到每遍的 4096 位响应；9 遍逐位多数投票得 w。
2. `data = np.packbits(w, bitorder="big")`（512 字节）；`ecc = bchlib.BCH(t, m=13).encode(data)`。
3. 用操作系统随机数生成 544 字节的 Toeplitz 种子 `seed`。
4. 计算根 r（32 字节）：`x = np.unpackbits(data, bitorder="little")`，`s = np.unpackbits(seed, bitorder="little")`，`rows = sliding_window_view(s, 4096)[:256]`，`r = np.packbits((rows @ x) % 2, bitorder="little")`。
5. 登记数据 `enrollment.json` 含 `device_id`、`lanes`、`m`、`t`、`ecc_hex`、`seed_hex`、`challenges_sha256`（即上表 K 的 SHA-256）。令 `helper = json.dumps(这些字段, sort_keys=True, separators=(",", ":")).encode()`，计算标签 `tag = HMAC-SHA256(r, b"BoundKV-tag|" + helper)`，写入 `tag_hex` 字段。
6. 只输出 `enrollment.json`（公开数据），不输出 w、r 或任何一遍原始响应。

**部署模式**：主机每次发一行 JSON 请求，板子回一行 JSON。

| 请求 | 板上动作 | 成功回复 |
|---|---|---|
| `{"op": "unlock", "passphrase_hex": "..."}` | 见下 | `{"success": true, "device_id": "...", "timing": {"acquire_s": ..., "fe_s": ..., "kdf_s": ...}}` |
| `{"op": "derive", "label_hex": "..."}` | `key = HMAC-SHA256(k_u, label)` | `{"success": true, "device_id": "...", "key_hex": "<64 位十六进制>"}` |
| `{"op": "lock"}` | 把 k_u 覆写为 0 并丢弃 | `{"success": true, "device_id": "..."}` |

unlock 的步骤：

1. 对 K 读 **1 遍**，按 `lanes` 组合得 w'（计时 `acquire_s`）；
2. 用 `ecc` 和 BCH(t, m=13) 纠错得 w，按登记步骤 4 算 r；由 `enrollment.json` 重新算 `helper` 和标签，与 `tag_hex` 比较（计时 `fe_s`）。纠错失败或标签不符时回复 `{"success": false, "device_id": "...", "error": "reconstruction"}`；
3. `z = HMAC-SHA256(r, b"BoundKV-unlock|" + passphrase)`，再重复 c 次 `z = HMAC-SHA256(r, z)`，`k_u = z`（计时 `kdf_s`）；
4. 把 r、w、w' 覆写为 0。

在 unlock 之前收到 derive 时，回复 `{"success": false, "device_id": "...", "error": "locked"}`。

部署模式的约束：

- r、k_u、w、w' 不写入文件、日志或串口输出，也没有读出它们的命令。
- 部署模式只提供上表三种请求，不接受研究模式的任何命令，不评估 K 以外的挑战。
- K 固化在部署固件中；若从文件读取，评估前先核对其 SHA-256，不符则不评估任何挑战。
- 纠错失败与标签不符返回同一个 `reconstruction` 错误，回复中不区分两者。
- 登记模式与部署模式分属两个固件映像；板子交付部署后，不再运行登记模式。
- 建议把 r 和 k_u 只放在 Zynq 的片上存储（OCM）；做不到时，在说明中写明实际存放位置。

**Z5 迭代次数标定**：在每块板上标定 c，使 unlock 的 `kdf_s` 不少于 1.0 s；两块板用同一个 c。记录 c 和实测的 `kdf_s`。

**Z6 主机客户端**：提供一个客户端程序：从 stdin 读一行请求 JSON，转发给板子，再把板子的回复原样打印到 stdout，退出码为 0。链路（UART、USB 或以太网）由硬件方选定，并在说明中写明。J6 按以下配置调用它：

```json
{"backend": "hardware", "device_id": "<板序列号>", "command": ["python3", "module_client.py", "--port", "..."], "timeout_seconds": 60}
```

**Z7 解锁耗时**：每块板上电后连续 unlock 20 次（正确口令），记录每次回复中的 `timing`；主机端另测 20 次 derive 的往返耗时。

**Z8 接口关闭**：部署模式下，向每块板发送 100 条研究模式的读取命令，挑战取自 Z3 的挑战集（不在 K 内），全部应被拒绝；同时记录一次 unlock 是否仍然正常。

### P1（有余力再做）

- **Z9 更多板子**：若能再借到 Zybo Z7-10，烧同一 bitstream，用研究模式对 K 读 9 遍，按 lane 输出 `apuf_lanes` 和 `ff_lanes`，并记录板序列号。

---

## U. Alveo U200 DDR4 + DRAM Bender

**原理**：在固定的 DRAM 段上以低于规格的 tRCD 读取，出现失效的单元由制造差异决定（Kim et al., "The DRAM Latency PUF", HPCA 2018）。

| 编号 | 内容 |
|---|---|
| U0 | **环境**：在 U200 上部署 DRAM Bender。记录 DIMM 型号、SPD 序列号、rank 数、每个 rank 的芯片数和位宽、**DQ 到芯片的映射**、温度的读取方式。冒烟测试：同一段在额定 tRCD 下零失效，在降低的 tRCD 下出现失效 |
| U1 | **参数（确定一次，之后不变）**：S = 4 个固定段，每段 8 KiB，分布在不同 bank（给出 bank/row/column）；一个固定的写入图样；降低后的 tRCD 取一个值，使室温下每段在 20 次读取中至少失效一次的单元占 1%–10%；其余时序为额定值 |
| U2 | **断电采集**：一条 DIMM，至少 10 次**真断电**，即主机断电不少于 60 s。每次上电后，每段读 N = 20 次：每次先用额定时序写入图样，再用降低的 tRCD 读出，记录逐单元的失效位图 |
| U3 | **计时**：每段一次评估（写入加 N 次读取）的耗时，记入 `boot.json` |
| U4 | **卫生**：评估后，该段用额定时序重写；说明降时序模式如何只对可信逻辑开放 |
| U5（P1） | 用同型号的第二条 DIMM 重复 U2 |

---

## 交付

交付推送到本仓库根目录下的 `delivery_YYYYMMDD/`，用采集日期命名；不要修改 `req_1120/` 和 `jetson_package/`，也不要覆盖旧交付。三部分可以分批交付，每批一个新目录，只放该批的子目录。

```
delivery_YYYYMMDD/
  jetson/
      env/        # J0 各命令输出、pip freeze
      tests/      # triton_smoke.json、pytest 输出
      results/    # J1–J5 的输出 JSON、J4 probe 输出
      binding/    # J6 输出 JSON、四个模块配置（不含口令）
  zybo/
      boards.json                                   # Z1
      temperature/<板序列号>_<room|60c>_apuf_lanes.npy   int8, shape (20, 4096, 4)  # (上电次数, K 中的挑战, lane)
      temperature/<板序列号>_<room|60c>_ff_lanes.npy     int8, shape (20, 4096, 4)
      temperature/<板序列号>_<room|60c>_meta.json        # Z2 每次上电的记录
      crp_board2/challenges.npy    int8, shape (N, 64)
      crp_board2/apuf_lanes.npy    int8, shape (N, 4)
      crp_board2/ff_lanes.npy      int8, shape (N, 4)
      crp_board2/meta.json         # 板序列号、生成种子、去重后的 N、XADC 温度
      module/     # Z4–Z6 的源码、两块板的 enrollment.json、calibration.json（c 与 kdf_s）、链路说明
      timing/     # unlock_timing_<板序列号>.json（Z7）
      interface/  # Z8 日志
      boards_extra/<板序列号>_apuf_lanes.npy、_ff_lanes.npy   int8, shape (9, 4096, 4)   # Z9（如有）
  u200/
      env/        # env.json、params.json、冒烟日志
      <DIMM SPD 序列号>/boot_01/ … boot_10/
          fail_bitmaps.npy   uint8, shape (4, 20, 8192)   # (段, 读取次数, np.packbits 后的 65536 位)，1 = 失效
          boot.json
  logs/           # 完整终端日志
  requirements_status.json
  SHA256SUMS.txt
```

- 数组只放 `.npy`，元数据只放 JSON；原始数据不做任何后处理，例如多数投票、XOR 或筛选。
- `SHA256SUMS.txt` 中的路径相对于交付目录本身，在交付目录内运行 `sha256sum -c SHA256SUMS.txt` 必须全部通过。
- `requirements_status.json` 按本文的编号（J0–J6、Z1–Z9、U0–U5）逐项写 `done / partial / blocked / not_started`，并说明缺口。
- 未做的条目如实标注，不要用模拟数据替代。任何"接口关闭"的声明，都要写明依据：是接口本身的限制、配置回读，还是仅为流程约定。
- 不要提交口令、r、k_u 或任何一遍未经处理的部署模式响应。

---

## 时间（建议）

| 时间 | 内容 |
|---|---|
| 第 1 周 | Z1、Z2（最先）、Z3；J0–J5；U0、U1 |
| 收到 Z2 后我方给出登记参数，此后 2 周 | Z4–Z8 |
| 此后 1 周 | J6；U2–U4 |
| 有余力时 | Z9、U5 |
