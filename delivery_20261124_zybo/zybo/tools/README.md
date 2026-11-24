# 工具

主机端脚本（Python 3.10+，numpy）通过 SSH 驱动板子；板端脚本 `zybo_board.py` 是 Python 3.6 / numpy 1.13。

| 文件 | 运行在 | 用途 |
|---|---|---|
| `zybo_board.py` | 板子 | 通过冻结的研究服务（127.0.0.1:3040）读原始 lane：`snapshot`、`read`、`sweep`、`load`。自己不碰 PL 寄存器 |
| `collect_sameboot.py` | PC | 同开机预实验：上传与校验、idle_pre、crp_board1、sweep、idle_post、快照、校验 |
| `record_boards.py` | PC | Z1：记录每块可达的板子的标识，写 `zybo/boards.json` |
| `run_module_board_tests.py` | PC | 根模块在实板上的验证：prepare、calibrate、live_acquire、matrix、timing、closure、j6_layout、ocm_check、cleanup、final |
| `power_cycle_campaign.py` | PC | Z2：断电采集（人工或可开关插座），无效轮次判定，凑够后 `assemble`；`--steps rehearsal` 在不断电的情况下演练 |
| `test_power_cycle.py` | PC | 断电工具的状态机测试（假板子、假时钟），8 项 |
| `verify_delivery.py` | PC | 交付目录的格式、哈希、完整性检查，以及输出中不含密钥、口令、根 |
| `make_timeline.py` | PC | 把 `logs/<run>/commands.jsonl` 整理成 `timeline.json`（和 `run_status.json`） |
| `host_common.py` | PC | 共用：SSH/SCP 封装，每次调用记入 `commands.jsonl` |

每个主机端脚本都把经手的所有远端命令写进对应 `logs/<run>/commands.jsonl`。`zybo/tools/` 里是最终版本：批次进行中修过几处主机端脚本的问题（重新上传前先删除只读文件、步骤状态文件跨次合并、cleanup 列表命令的参数个数），都不涉及板端读取路径；每一步实际执行的命令以 `commands.jsonl` 为准。各步运行时板端脚本的版本见交付目录的 README。
