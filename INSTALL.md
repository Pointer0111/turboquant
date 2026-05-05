# AutoDL 安装指南（A800）

## 前置：配置 pip 源

**不要开学术加速**，`/etc/network_turbo` 会干扰镜像下载导致 `Connection reset`。如果开过，先关：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

换清华源，稳定性优于阿里源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip config set global.timeout 600
pip config set global.retries 10
```

如果清华源仍然失败，换中科大源：

```bash
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
pip config set global.trusted-host pypi.mirrors.ustc.edu.cn
```

---

## 第一步：检查现有环境

```bash
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
nvcc --version | head -1
python --version
```

---

## 第二步：安装 vLLM

```bash
pip install --no-cache-dir "vllm==0.20.1"
```

vLLM 0.20.1 会自动拉取 `torch==2.11.0` 及其 NVIDIA 依赖。配好清华源后应该可以直接跑通。

---

## 第三步：安装 TurboQuant

```bash
cd /root/turboquant   # 替换为你的项目路径
pip install -e .
```

---

## 第四步：验证

```bash
# 验证 vLLM
python -c "import vllm; print('vllm:', vllm.__version__)"

# 验证 TurboQuant 核心数学（不需要 GPU）
python - <<'EOF'
import torch
from turboquant.quantizer import TurboQuantMSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = 128
tq = TurboQuantMSE(dim=dim, bits=3).to(device)
x = torch.randn(4, 8, dim, device=device)
x = x / x.norm(dim=-1, keepdim=True)
packed = tq.quantize(x)
print("quantize OK, packed shape:", packed.indices.shape)
print("TurboQuant import OK")
EOF
```

---

## 第五步：运行 proof.py

```bash
MODEL=/root/autodl-fs/Qwen3.5-27B TP=1 python proof.py
```

---

## 备用：wget 手动下载（pip 仍然超时时使用）

如果换源后 pip 还是卡在某个大文件，从报错信息里复制 URL，用 wget 手动下载再本地安装：

```bash
# URL 从 pip 报错信息里复制
wget -c --retry-connrefused --waitretry=5 -t 0 "<URL>" -O /tmp/<文件名完整>.whl
pip install --no-cache-dir /tmp/<文件名完整>.whl
```

`-c` 断点续传，`-t 0` 无限重试，断了直接重跑同一条命令。

**注意**：`-O` 后面的文件名必须和 URL 里的原始文件名一致（含完整平台标签，如 `torch-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl`），否则 pip 会报 `not a valid wheel filename`。

---

## 常见问题

**`Connection reset by peer`**：关掉学术加速（见前置步骤）。

**`not a valid wheel filename`**：`-O` 指定的文件名不完整，补全平台标签后重装。

**vLLM monkey-patch 报 AttributeError**：vLLM 0.20.1 与测试版本 0.18.0 内部 API 可能有差异，记录报错位置告知我。
