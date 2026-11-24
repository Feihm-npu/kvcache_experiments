# J6 的四个模块配置

格式同 `jetson_package/configs/module_hardware.example.json`。`module_client.py` 要在 J6 运行目录能被 `python3` 找到（放进该目录或 `PYTHONPATH`）。

| 配置 | 板子 | 板上运行的服务 | `device_id` |
|---|---|---|---|
| `owner.json` | 板 1，`192.168.1.2:3041` | 用板 1 自己的 `enrollment.json` 启动 `module_serve.py` | 板 1 序列号 `210251A08870`（已填好） |
| `other.board2.json` | 板 2，`<board 2 address>:3041` | 用板 2 自己的 `enrollment.json` | 板 2 序列号（待填） |
| `moved.board2.json` | 板 2，`<board 2 address>:3041` | 用**板 1 的** `enrollment.json` 启动 `module_serve.py` | 板 1 序列号：服务回复的是登记文件里的 `device_id` |
| `modified.board1.json` | 板 1，`192.168.1.2:3043` | 用板 1 的 `enrollment.json` 的一份副本启动第二个 `module_serve.py`，副本的 `ecc_hex` 改了 1 个字节；第二个实例要加 `--port 3043 --ocm-address 0xFFFE9000` | 板 1 序列号 |

`owner` 与 `modified` 可以同时运行在板 1 上（两个端口、两页 OCM），本批在实板上验证过这个布局（`../validation/j6_two_services.json`）：正常的实例 unlock 成功，`modified` 实例回 `reconstruction`。同一块板上不要让两个实例同时 unlock，它们共用 PL 的寄存器。

四个配置都还没有正式的 `enrollment.json`：它们等 Z2 之后的 `lanes` 和 `t`。
