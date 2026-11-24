# Zybo 批次（2026-11-24）：板 1 标识、同开机预实验、根模块固件与实板验证

需求见 `../req_1120/README.md` 的 Z 部分。本批**没有完成任何一项 P0**：真断电 0 次，板 2 不在，正式登记没有做。做完并交付的是：板 1 的标识、一组同开机的原始数据、根模块（Z4–Z8）的全部固件与测试、这些固件在板 1 上的实测，以及断电采集和后续步骤要用的工具。

同一天的 Jetson 批次已占用 `delivery_20261124/`，所以本批放在 `delivery_20261124_zybo/`，日期仍是采集日期。

## 逐项状态

| 编号 | 状态 | 一句话 |
|---|---|---|
| Z1 | partial | 板 1 已标识；板 2、印字标签、Device DNA 缺 |
| Z2 | blocked | 真断电 0 次（无人值守，切不了电源，没有热源）；工具已备好并在实板上演练 |
| Z3 | blocked | 板 2 不在；同一挑战集在板 1 上读了补充数据 |
| Z4 | blocked | 等你们给 `lanes` 和 `t`；固件已实现并在实板验证 |
| Z5 | partial | 板 1：c = 4049，kdf_s ≥ 1.083 s；板 2 不在 |
| Z6 | partial | 客户端已交付并在实板回路中测过；只有一块板 |
| Z7 | partial | 临时测量（测试登记 + 回放）；正式测量等正式登记和上电 |
| Z8 | partial | 实板测过一次（测试登记 + 回放），全部拒绝、unlock 正常 |
| Z9 | not_started | 没有第三块板 |

说明和证据路径见 `requirements_status.json`。J0–J6、U0–U5 只是沿用或指向其他批次。

## 没做成的原因

1. **只有一块 Zybo。** USB 上只有一个 Digilent JTAG 设备，`192.168.1.0/24` 上只有一块板。
2. **真断电需要人在场或可控电源。** 板子的电源我切不了：主机上没有可开关的插座，USB 端口电源不可控，主机的 sudo 需要密码。复位、软重启和重新加载 bitstream 都不算真断电，所以一次都没有做。
3. **没有热源。** Z2 的"约 60 °C"要靠外部加热后在那个温度下上电。本批只做到了用 CPU 负载把芯片加热后读取（同开机），这是另一回事。
4. **登记参数还没给。** `lanes` 和 `t` 要由你们根据 Z2 数据定，所以 Z4–Z8 的正式部分都做不了。
5. **主机上的 dnsmasq 没有运行。** 板子一旦断电，重新上电后拿不到 `192.168.1.2`，只剩 PYNQ 镜像的静态地址 `192.168.2.99`。做 Z2 之前必须先解决这一点。

## 目录

```
zybo/
  boards.json                  Z1：板 1 的标识与依据
  sameboot_pilot/210251A08870/ 同开机补充数据（不是 Z2）：idle_pre、sweep、idle_post 各一组
  crp_board1/                  Z3 的挑战集在板 1 上的读数（补充，不是 Z3）
  module/                      Z4–Z8：固件源码、测试、板 1 的标定与验证输出、J6 的配置模板；说明见 module/README.md
  timing/                      Z7 的临时测量（文件名带 _replay_test）
  interface/                   Z8 的实板记录与服务日志
  tools/                       采集与验证工具
  env/                         各阶段的板子快照
logs/                          每次 ssh/scp 调用（commands.jsonl）、按步骤整理的时间线（timeline.json）和原始日志
requirements_status.json
SHA256SUMS.txt
```

## 数据

所有数组都是研究服务返回的原始 4 个 lane 字节（int8），没有投票、XOR 或筛选；服务多给的第 5 个 XOR 字节丢弃，没有计算。每个目录的 `meta.json` 记录每次读取前后的 XADC 温度、VCCINT、板端单调时钟与 uptime、`boot_id`、bitstream 与固件的 SHA-256。

| 目录 | 内容 | 读取时 XADC 温度 |
|---|---|---|
| `sameboot_pilot/…/idle_pre` | K × 20 遍（apuf_lanes、ff_lanes 各 20），间隔 5 s | 56.4–57.8 °C |
| `sameboot_pilot/…/sweep` | 芯片被两个最低优先级的 CPU 负载（npy 数组乘法）加热到约 66 °C 后再自然冷却期间的 228 遍 K：`under_load` 10 遍（负载运行，读取进程优先级更高）、`cool_hot` 60 遍、`cool` 158 遍 | 56.8–66.4 °C |
| `sameboot_pilot/…/idle_post` | 冷却回空闲后 K × 20 遍 | 56.1–57.3 °C |
| `crp_board1` | Z3 挑战集（`default_rng(20261120)`，去掉与 K 重复的行后 N = 100000）每条读 1 遍，两族各一个 `int8 (N, 4)` 数组 | 约 57.5 °C |

这组数据与 0917、1113 两批对得上：各 lane 的 P(1) 相同，单次读取相对空闲多数票的偏差 apuf 约 0.18–0.22%、ff 约 0.28–0.35%，66 °C 时比 57 °C 高约两成。

三点读取上的事实，供你们定协议时参考：

- XADC 的温度读数与 PS 当时的负载有关：sweep 里负载运行时读到 66.1–66.4 °C，负载停止并等 3 s 后是 64.1 °C，头几秒内还会继续下降。所以 sweep 中只有 `under_load` 的 10 遍是在负载运行时读的，其余都在负载停止 3 s 之后。
- 第一次扫描用固定温度步长触发读取，因为读数随负载跳动而反复触发，已中止，没有留下数据。
- 板子空闲时芯片就有 56–58 °C（同一开机内的漂移），不是室温。

## 根模块

见 `zybo/module/README.md`。要点：

- `module_enroll.py`（登记映像，每板一次）和 `module_serve.py`（部署映像，只有 unlock / derive / lock），共用 `module_core.py`；板上是 Python 3.6 / numpy 1.13 / bchlib 2.1.3。
- 32 项测试在 PC 和板子上各通过一次（`logs/module/tests_on_*.txt`）。
- 实板：真实读取路径约 1.06 s 一遍 K（两族）；fe_s 56–66 ms（t = 64）；c = 4049 时 kdf_s ≥ 1.083 s；r 与 k_u 常驻在 OCM 的一页里，服务停止后读回全零。
- 验证用的都是**测试登记**：由登记映像从本批已录下的 9 遍同开机读数算出，只喂给回放后端，用完即删，没有交付。板 1 上没有用临时参数跑过正式登记。
- `t ≤ 64`：`bchlib.BCH(t, m=13)` 的上限；宽 `lanes` 在同开机数据里就会超预算（`module/validation/reconstruction_matrix.json`）。

## 需要现场完成的事

**接上板 2 之后**

```bash
# 主机：给板子的网线口提供 DHCP（需要 sudo），否则板子上电后连不上
sudo nmcli connection up fpga-link                # enp4s0 = 192.168.1.1/24，已配置过则跳过
sudo dnsmasq -C /tmp/opencode/dnsmasq_fpga.conf   # 内容如下；板 2 要加一行 dhcp-host
#   interface=enp4s0
#   bind-interfaces
#   listen-address=192.168.1.1
#   port=0
#   dhcp-range=192.168.1.10,192.168.1.50,255.255.255.0,12h
#   dhcp-host=00:0a:35:00:1e:53,192.168.1.2
#   dhcp-option=option:router,192.168.1.1
#   dhcp-option=option:dns-server,192.168.1.1

# 标识两块板（填各自的序列号和地址；有人读过印字标签时加 --label-read）
python3 zybo/tools/record_boards.py --out <新交付目录>/zybo/boards.json --logs <新交付目录>/logs/boards \
    --board 1=210251A08870=xilinx@192.168.1.2 --board 2=<板 2 序列号>=xilinx@192.168.1.3
```

两块板的 MAC 可能相同（都是镜像的默认值），dnsmasq 的 dhcp-host 要按各自的地址分别绑定，或者在板 2 的 U-Boot 环境里改掉 MAC。

**Z2（每块板、每个温度各 20 次真断电）**

```bash
# 人在板旁按提示拔插电源（或用 --power switch --power-off-cmd .. --power-on-cmd .. 接可开关的插座）
python3 zybo/tools/power_cycle_campaign.py --out <新交付目录> --serial 210251A08870 --condition room --rounds 20
python3 zybo/tools/power_cycle_campaign.py --out <新交付目录> --serial 210251A08870 --condition 60c  --rounds 20
# 60c：先把板子加热并在约 60 °C 下保持，再按提示断电、上电；每次读取的 XADC 温度都会记录
```

工具自己保证断电不少于 60 s；不足 60 s、`boot_id` 没变或上电后跑的不是冻结服务（带 PL 配置）的轮次判为无效并留档。凑够 20 个有效轮次后用 `--steps assemble` 生成 `zybo/temperature/` 下的数组。

**Z3**：板 2 上传 Z3 挑战集后用板端的 `zybo_board.py read --challenges <文件> --repeats 1` 读取，把两个数组（去掉只有 1 遍的那一维，成为 `int8 (N, 4)`）和 `meta.json` 放进 `zybo/crp_board2/`；`collect_sameboot.py` 的 `crp_board1` 步骤是同样做法的主机端封装。

**Z4–Z8**：收到 `lanes` 和 `t` 之后按 `zybo/module/README.md` 的最后一节在每块板上登记一次、启动部署服务，然后 Z7、Z8 各做一遍（都是上电之后）。

## 校验

```bash
cd delivery_20261124_zybo
sha256sum -c SHA256SUMS.txt
python3 zybo/tools/verify_delivery.py . --sha                 # 需要 numpy
python3 -m unittest discover -s zybo/module -p "test_*.py"    # 32 项，需要 numpy 和 bchlib
python3 -m unittest discover -s zybo/tools  -p "test_*.py"    # 8 项，只需要 numpy
```

## 工具版本

采集脚本 `zybo/tools/zybo_board.py` 在批次中改过几次，读取路径没有变，改的是扫描的控制逻辑和元数据字段。各步实际运行的版本（SHA-256）：`idle_pre`、`crp_board1`、`snapshot_start` 用 `b0c81749…`；`sweep` 用 `37df1b86…`；`idle_post`、`snapshot_end` 和断电演练用 `b1df0e72…`，即交付的这份。前两版与交付版的差别：扫描第二版、停止文件的清理、`meta.json` 里的 `tool` 字段。

## 板子现状

板 1 没有断电、没有重启（`boot_id` 未变，uptime 连续）。PL 在本批中被重新配置过一次（断电演练，用冻结 bitstream）。研究服务被重启过两次，现在运行的启动参数与原来相同（从冻结 bitstream 配置 PL）；根模块服务都已停止，测试登记与回放文件已从板上删除，两页 OCM 读回全零。板上的 `/home/xilinx/puf/req1120_20261124/` 留有工具、模块源码和原始读数，供后续批次使用。
