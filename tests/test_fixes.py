"""
P1 正确性修复验收测试（§4：1~7、9、10、11）

覆盖第 1~7、9、10 条，以及第 11 条的前半（生成路径与训练路径同源）。
第 8 条（audio target 来自真实 codes 的可学性）与第 11 条后半（真实音频
闭环 wav）需要 Encodec 码本与训练，见 tests/overfit_audio_small.py 与
tests/closed_loop_audio.py。

    python tests/test_fixes.py              # 直接运行（不依赖 pytest）
    python -m pytest tests/test_fixes.py    # 装了 pytest 也能收集
"""

import math
import sys
import shutil
import tempfile
import uuid
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asm_ersa import ASM_ERSA_Layer, ERSA_Omni, ASM_ERSA_Omni, Modality


# ═══════════════════════════════════════════════════════════════
# 小模型配置
# ═══════════════════════════════════════════════════════════════

# n_layers=4 → 按 asm_ersa_omni.py 的 is_ersa = (i % 4 == 3)，恰好 1 个 ERSA 层（idx 3）
TINY = dict(
    d_model=64, n_heads=2, d_k=16, d_v=16, d_ff=128, n_layers=4,
    vocab_size=128, max_seq_len=2048,
    n_shared_experts=2, n_text_experts=2, n_audio_experts=2, n_vision_experts=2,
    top_k=2, n_audio_tokens=64,
    audio_config={'n_mels': 16, 'patch_size': 4, 'stride': 2},
    vision_config={'image_size': 28, 'patch_size': 14, 'n_layers': 1},
)
N_LAYERS = TINY['n_layers']
N_ERSA_LAYER = 3                # 唯一的 ERSA 层下标
N_ASM_LAYERS = 3                # 其余是 ASM-ERSA 层
D_MODEL, N_HEADS, D_K, D_V = 64, 2, 16, 16
N_SCALES = 4                    # ASM-ERSA层默认 MSTD 尺度数
N_SLOTS = 512                   # ASM_ERSA_Layer 默认槽位数
MEL_FRAMES, AUDIO_VALID, N_MELS = 200, 140, 16

# 沙箱/CI 下系统临时目录可能不可写：测试自建仓库内临时根（.cache/ 已被 gitignore）
TMP_ROOT = Path(__file__).resolve().parent.parent / ".cache"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# 编码器卷积参数（AudioEncoder.conv[0]）
CONV_K, CONV_S = 4, 2
N_AUDIO_TOK = ASM_ERSA_Omni.audio_token_len(MEL_FRAMES, CONV_K, CONV_S)   # 51
N_AUDIO_CLEAN = ASM_ERSA_Omni.audio_valid_len(AUDIO_VALID, CONV_K, CONV_S)  # 35


def build_model(seed: int = 0) -> ASM_ERSA_Omni:
    torch.manual_seed(seed)
    model = ASM_ERSA_Omni(**TINY)
    model.eval()
    return model


@contextmanager
def expect_raises(exc_type):
    """with expect_raises(AssertionError) as box: ...; assert 'x' in str(box.value)"""
    box = SimpleNamespace(value=None)
    try:
        yield box
    except exc_type as e:
        box.value = e
        return
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但没有抛出")


def rand_ids(B: int, T: int) -> torch.Tensor:
    return torch.randint(0, TINY['vocab_size'], (B, T))


def rand_mel(B: int, frames: int = MEL_FRAMES) -> torch.Tensor:
    return torch.randn(B, N_MELS, frames)


# ═══════════════════════════════════════════════════════════════
# §4-1 状态确实更新
# ═══════════════════════════════════════════════════════════════

def test_state_updates_and_persists():
    model = build_model()
    model.train()
    B, T = 2, 8
    ids = rand_ids(B, T)

    out1 = model(input_ids=ids, return_states=True)
    states1 = out1['states']
    assert len(states1) == N_LAYERS
    assert states1[N_ERSA_LAYER] is None, "ERSA 层不应有状态"

    # ASM-ERSA 状态是 (K, V, beta) 三元组：[B, H, M, d_k] / [B, H, M, d_v] / [B, H, M]
    for li in range(N_ASM_LAYERS):
        s = states1[li]
        assert isinstance(s, tuple) and len(s) == 3, \
            f"层 {li} 状态不是 (K,V,beta)：{type(s)}"
        K, V, beta = s
        assert tuple(K.shape) == (B, N_HEADS, N_SLOTS, D_K), \
            f"层 {li} K 形状 {tuple(K.shape)}"
        assert tuple(V.shape) == (B, N_HEADS, N_SLOTS, D_V), \
            f"层 {li} V 形状 {tuple(V.shape)}"
        assert tuple(beta.shape) == (B, N_HEADS, N_SLOTS), \
            f"层 {li} beta 形状 {tuple(beta.shape)}"
        assert K.dtype == torch.float32 and V.dtype == torch.float32, \
            f"层 {li} 状态未按 fp32 累积"
        assert float(K.detach().abs().sum()) > 0, f"层 {li} K 状态恒为 0：序列结束后没有被写入"
        assert float(V.detach().abs().sum()) > 0, f"层 {li} V 状态恒为 0：序列结束后没有被写入"

    # 确定性对照：同输入、同起点 → 逐位相同。不成立则下面的差异断言无意义
    out1b = model(input_ids=ids)
    assert torch.equal(out1b['text_logits'], out1['text_logits']), \
        "同输入两次 forward 结果不同，模型存在随机性"

    # 把状态传回去 → 输出必须改变（状态参与计算且跨调用延续）
    out2 = model(input_ids=ids, states=states1, return_states=True)
    assert not torch.allclose(out2['hidden_states'], out1['hidden_states']), \
        "传回状态后输出没变：状态没有参与计算（A1 类缺陷）"
    assert not torch.allclose(out2['states'][0][1], states1[0][1]), "状态没有继续演化"

    # 同一个状态传两次 → 结果可复现（状态是显式输入，不藏在模块内部）
    out3 = model(input_ids=ids, states=states1)
    assert torch.equal(out3['text_logits'], out2['text_logits']), \
        "传入相同状态得到不同结果，状态语义不确定"


# ═══════════════════════════════════════════════════════════════
# §4-2 存在 BPTT
# ═══════════════════════════════════════════════════════════════

def test_bptt_through_state():
    """状态递推必须留在计算图上：末位 loss 的梯度能回到第 0 个 token。

    对照组（每步 detach）梯度必须为 0，否则说明这条断言不是靠状态路径成立的。
    """
    torch.manual_seed(0)
    d_model, T = 16, 6
    layer = ASM_ERSA_Layer(d_model, 2, d_k=8, d_v=8, n_scales=2)
    layer.train()

    def first_token_grad(detach_state: bool) -> float:
        x = torch.randn(1, T, d_model, requires_grad=True)
        state, outs = None, []
        for t in range(T):
            o, state = layer(x[:, t], state=state)
            outs.append(o)
            if detach_state:
                state = tuple(s.detach() for s in state)
        outs[-1].sum().backward()
        return float(x.grad[:, 0].abs().max())

    g_live = first_token_grad(detach_state=False)
    g_cut = first_token_grad(detach_state=True)
    assert g_live > 0, "梯度传不回第 0 个 token：状态递归断在计算图外（无 BPTT）"
    assert g_cut == 0.0, "对照组（逐步 detach）竟有梯度，测试本身不成立"


def test_bptt_full_model():
    model = build_model()
    model.train()
    ids = rand_ids(1, 8)
    out = model(input_ids=ids)
    model.zero_grad(set_to_none=True)
    out['text_logits'][:, -1].sum().backward()

    grad = model.text_embed.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0
    assert float(grad[ids[0, 0]].abs().max()) > 0, "梯度没有回到第 0 个 token 的嵌入"


# ═══════════════════════════════════════════════════════════════
# §4-3 无跨样本污染
# ═══════════════════════════════════════════════════════════════

def test_no_cross_sample_contamination():
    model = build_model()
    ids = rand_ids(3, 10)
    mel = rand_mel(3)
    valid = torch.tensor([MEL_FRAMES, AUDIO_VALID, MEL_FRAMES])

    cases = [
        ('text', dict(input_ids=ids)),
        ('audio+text', dict(input_ids=ids, audio_mel=mel, audio_valid_frames=valid)),
    ]
    for name, kwargs in cases:
        full = model(**kwargs)['hidden_states']
        for b in range(3):
            row = {k: v[b:b + 1] for k, v in kwargs.items()}
            single = model(**row)['hidden_states']
            diff = float((full[b:b + 1] - single).detach().abs().max())
            assert diff < 1e-5, \
                f"[{name}] 样本 {b} 的 batch/单样本输出差 {diff:.3e}：存在跨样本污染"


# ═══════════════════════════════════════════════════════════════
# §4-4 因果性黄金测试
# ═══════════════════════════════════════════════════════════════

def test_causality_golden_text():
    model = build_model()
    B, T = 2, 12
    ids = rand_ids(B, T)
    base = model(input_ids=ids)['hidden_states']

    for t in (3, 7, 10):
        pert = ids.clone()
        pert[:, t + 1:] = rand_ids(B, T - t - 1)
        assert not torch.equal(pert, ids)

        out = model(input_ids=pert)['hidden_states']
        assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-6, rtol=0), \
            f"位置 <= {t} 的输出被未来输入改变：存在因果泄漏（A3 类缺陷）"
        assert not torch.allclose(out[:, t + 1:], base[:, t + 1:], atol=1e-6, rtol=0), \
            f"t={t} 的扰动没有产生任何影响，该断言不足以证明因果性"


def test_causality_golden_cross_modal():
    """未来的 TEXT 不得影响音频前缀与更早的文本位置（防标签泄漏）。"""
    model = build_model()
    B, T_text = 2, 10
    mel = rand_mel(B)
    ids = rand_ids(B, T_text)

    base = model(audio_mel=mel, input_ids=ids)['hidden_states']
    assert base.shape[1] == N_AUDIO_TOK + T_text

    t = 4
    pert = ids.clone()
    pert[:, t:] = rand_ids(B, T_text - t)
    out = model(audio_mel=mel, input_ids=pert)['hidden_states']

    keep = N_AUDIO_TOK + t          # 全部音频前缀 + 位置 < t 的文本
    assert torch.allclose(out[:, :keep], base[:, :keep], atol=1e-6, rtol=0), \
        "未来文本影响了音频前缀或更早的文本位置"


# ═══════════════════════════════════════════════════════════════
# §4-5 mask 语义
# ═══════════════════════════════════════════════════════════════

def test_causal_mask_rules():
    ersa = ERSA_Omni(d_model=8, n_heads=1, d_k=4, d_v=4, max_seq_len=64)
    A, TXT, V = int(Modality.AUDIO), int(Modality.TEXT), int(Modality.VISION)

    def row(mods, n_prefix, q=None):
        m = torch.tensor([mods])
        return ersa._build_causal_mask(m, n_prefix, q).tolist()[0]

    # 规则 2：前缀内同模态双向 —— 音频查询看得到它之后的音频
    assert row([A, A, A, A], 4, 0) == [True, True, True, True]
    # 规则 2 的边界：跨模态不双向 —— 音频查询看不到前缀里的 VISION
    assert row([A, V, V, V], 4, 0) == [True, False, False, False]
    # 规则 3：任何 token 不得看到未来的 TEXT
    assert row([A, A, TXT, TXT], 2, 0) == [True, True, False, False]
    assert row([A, A, TXT, TXT], 2, 1) == [True, True, False, False]
    # 规则 1：TEXT 严格因果
    assert row([TXT, TXT, TXT, TXT], 0, 0) == [True, False, False, False]
    assert row([TXT, TXT, TXT, TXT], 0, 1) == [True, True, False, False]
    # 规则 4：文本查询能看到之前的前缀
    assert row([A, A, TXT, TXT], 2, 2) == [True, True, True, False]

    # 线上调用（query_index=None）查询恒为 buffer 末位，基础行 j <= i 恒真：
    # 此时 mask 不再额外限制，规则 1/3/4 由 buffer 长度结构性保证，
    # 规则 2 的双向性要等并行前缀前向才生效。
    assert row([A, A, TXT, TXT], 2) == [True, True, True, True]


# ═══════════════════════════════════════════════════════════════
# §4-6 打分符号（无 ReLU）
# ═══════════════════════════════════════════════════════════════

def test_sparse_scores_signed():
    dk = 4
    k = torch.randn(1, 1, dk)
    K = torch.stack([k, -k, torch.randn(1, 1, dk) * 0.1], dim=1)   # [B, 3, H, dk]

    scores = ERSA_Omni._sparse_scores(k, K, dk)
    assert tuple(scores.shape) == (1, 3)
    assert float(scores[0, 1]) < 0, "q = -k 的打分被截成 0：ReLU 没有移除"
    assert int(scores.argmax(dim=-1)[0]) == 0, "负相关项排到了 top-1"

    # ERSA-Omni 的精确重打分层同样保留负分（不做 ReLU），前向应保持有限
    ersa = ERSA_Omni(d_model=8, n_heads=1, d_k=4, d_v=4, max_seq_len=64)
    out = ersa(torch.randn(1, 8), torch.randn(1, 5, 8),
               torch.arange(5).unsqueeze(0))
    assert torch.isfinite(out).all(), "ERSA-Omni 前向出现非有限值"


# ═══════════════════════════════════════════════════════════════
# §4-7 长序列容量
# ═══════════════════════════════════════════════════════════════

def test_ersa_long_sequence():
    max_len = 4096
    ersa = ERSA_Omni(d_model=16, n_heads=2, d_k=8, d_v=8,
                     block_size=64, top_blocks=4, window=64,
                     max_seq_len=max_len)

    def forward(t):
        mods = torch.full((1, t), int(Modality.TEXT), dtype=torch.long)
        return ersa(torch.randn(1, 16), torch.randn(1, t, 16),
                    torch.arange(t).unsqueeze(0), modality_ids=mods, n_prefix=0)

    # 块级 sketch 路由 + 精确重打分在长序列上必须能前向
    for t in (4, 2049, 2050, max_len):
        assert tuple(forward(t).shape) == (1, 16), f"t={t} 前向失败"

    # 超过 max_seq_len 必须显式报错，而不是静默截断
    with expect_raises(ValueError) as box:
        forward(max_len + 2)
    msg = str(box.value)
    assert f'max_seq_len={max_len}' in msg, f"越界报错信息不明确: {msg}"


# ═══════════════════════════════════════════════════════════════
# B8：audio_token_len / audio_valid_len 与编码器真实感受野一致
# ═══════════════════════════════════════════════════════════════

def test_audio_valid_len_matches_encoder():
    model = build_model()
    enc = model.audio_encoder

    mel = rand_mel(1)
    base = enc(mel)
    assert base.shape[1] == N_AUDIO_TOK, \
        f"audio_token_len 算出 {N_AUDIO_TOK}，编码器实际输出 {base.shape[1]}"

    probe = mel.clone()
    probe[0, :, AUDIO_VALID:] = torch.randn_like(probe[0, :, AUDIO_VALID:])
    changed = (base - enc(probe)).abs().amax(dim=(0, 2)) > 1e-6   # [T_tokens]

    assert not bool(changed[:N_AUDIO_CLEAN].any()), \
        f"前 {N_AUDIO_CLEAN} 个 token 竟然被 padding 内容改变：audio_valid_len 偏大"
    assert bool(changed[N_AUDIO_CLEAN:].all()), \
        f"第 {N_AUDIO_CLEAN} 个之后的 token 理应依赖 padding：audio_valid_len 偏小"


# ═══════════════════════════════════════════════════════════════
# §4-9 padding 屏蔽
# ═══════════════════════════════════════════════════════════════

def test_padding_invariance_model():
    model = build_model()
    B, T_text = 2, 8
    ids = rand_ids(B, T_text)
    text_mask = torch.ones(B, T_text, dtype=torch.bool)
    text_mask[1, 5:] = False                    # 右 padding
    mel = rand_mel(B)
    valid = torch.tensor([MEL_FRAMES, AUDIO_VALID])

    def run(mel_):
        return model(input_ids=ids, audio_mel=mel_,
                     attention_mask=text_mask, audio_valid_frames=valid)

    mel_noise = mel.clone()
    mel_noise[1, :, AUDIO_VALID:] = torch.randn_like(mel_noise[1, :, AUDIO_VALID:])
    a, b = run(mel), run(mel_noise)

    # padding_mask 必须如实反映两段各自的 padding
    pm = a['padding_mask']
    assert tuple(pm.shape) == (B, N_AUDIO_TOK + T_text)
    assert bool(pm[0, :N_AUDIO_TOK].all()), "样本 0 无 padding，音频 token 应全为真实"
    assert int(pm[1, :N_AUDIO_TOK].sum()) == N_AUDIO_CLEAN, \
        f"样本 1 的真实音频 token 数为 {int(pm[1, :N_AUDIO_TOK].sum())}，应为 {N_AUDIO_CLEAN}"
    assert bool(pm[1, N_AUDIO_TOK:].equal(text_mask[1])), "文本段 mask 与 attention_mask 不一致"

    # 有效音频 token（位于 pad 之前）
    assert torch.allclose(a['hidden_states'][1, :N_AUDIO_CLEAN],
                          b['hidden_states'][1, :N_AUDIO_CLEAN], atol=1e-6, rtol=0), \
        "mel 尾部 padding 的内容影响了有效音频 token"

    # 文本段在音频 pad 之后：pad 内容不得顺着 ASM-ERSA状态污染它
    assert torch.allclose(a['hidden_states'][1, N_AUDIO_TOK:][text_mask[1]],
                          b['hidden_states'][1, N_AUDIO_TOK:][text_mask[1]],
                          atol=1e-6, rtol=0), \
        "音频 padding 的内容污染了其后的文本 token"

    # 样本 0 完全不受样本 1 的 padding 影响
    assert torch.allclose(a['hidden_states'][0], b['hidden_states'][0], atol=1e-6, rtol=0)


def test_padding_invariance_loss():
    from train_omni_3b import TrainConfig, compute_loss

    model = build_model()
    model.train()
    cfg = TrainConfig(vocab_size=TINY['vocab_size'],
                      n_audio_tokens=TINY['n_audio_tokens'],
                      image_size=TINY['vision_config']['image_size'])

    B, T_text = 2, 8
    T_frames = 12
    ids = rand_ids(B, T_text)
    text_mask = torch.ones(B, T_text, dtype=torch.long)
    text_mask[1, 5:] = 0
    mel = rand_mel(B)

    # target 来自 batch（P2 起 loss 不再自己造随机占位）。样本 1 只有前 8 帧有效，
    # 尾部按 collate 约定补齐并被 codes_valid 屏蔽。
    torch.manual_seed(11)
    codes = torch.randint(0, TINY['n_audio_tokens'], (B, T_frames))
    codes_valid = torch.ones(B, T_frames, dtype=torch.bool)
    codes_valid[1, 8:] = False

    batch = {
        'input_ids': ids,
        'attention_mask': text_mask,
        'audio_mel': mel,
        'audio_valid_frames': torch.tensor([MEL_FRAMES, AUDIO_VALID]),
        'audio_codes': codes,
        'audio_codes_valid': codes_valid,
    }
    # 噪声同时打进 mel padding 与 code padding：两处都必须与 loss 无关
    noisy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    noisy['audio_mel'][1, :, AUDIO_VALID:] = torch.randn_like(
        noisy['audio_mel'][1, :, AUDIO_VALID:])
    noisy['audio_codes'][1, 8:] = torch.randint(
        0, TINY['n_audio_tokens'], (T_frames - 8,))

    la = compute_loss(model, batch, cfg, torch.device('cpu'))
    lb = compute_loss(model, noisy, cfg, torch.device('cpu'))

    assert 'audio_loss' in la, "音频离散模式下没有计算 audio_loss"
    for key in ('text_loss', 'audio_loss', 'total_loss'):
        va, vb = float(la[key].detach()), float(lb[key].detach())
        assert math.isfinite(va)
        assert abs(va - vb) < 1e-6, \
            f"padding 内容改变了 {key}: {va:.6f} vs {vb:.6f}"


# ═══════════════════════════════════════════════════════════════
# §4-10 存档往返
# ═══════════════════════════════════════════════════════════════

def test_save_load_roundtrip():
    model = build_model(seed=3)
    ids = rand_ids(2, 9)
    with torch.no_grad():
        before = model(input_ids=ids)['text_logits']

    # 不用 TemporaryDirectory：它在 Windows 上建成 0o700，本仓库沙箱下 safetensors
    # 的原子写（.tmp 临时文件 + rename）会被拒绝访问；改用普通 mkdir 目录。
    d = TMP_ROOT / ("save_" + uuid.uuid4().hex[:8])
    d.mkdir()
    try:
        model.save_pretrained(d)
        assert (Path(d) / 'model.safetensors').is_file()
        assert (Path(d) / 'config.json').is_file()
        loaded = ASM_ERSA_Omni.from_pretrained(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    assert loaded._init_kwargs == model._init_kwargs, "结构参数往返后不一致"
    loaded.eval()
    with torch.no_grad():
        after = loaded(input_ids=ids)['text_logits']
    assert torch.equal(before, after), "保存/加载往返后同输入输出不一致"


# ═══════════════════════════════════════════════════════════════
# §4-11（前半）生成路径与训练路径逐位对齐
# ═══════════════════════════════════════════════════════════════

def test_generate_audio_matches_teacher_forcing():
    """
    自回归生成（增量状态递推）与训练路径（整段前向 + teacher forcing）必须同源。

    用例 A 全帧有效时应当逐元素相等；用例 B 尾部 padding 时，有效位（key 未被屏蔽、
    也不受其后 pad 影响的位）同样应当相等——生成时永远没有 padding。
    """
    model = build_model(seed=5)
    B, T_text, T_frames = 2, 6, 7
    ids = rand_ids(B, T_text)
    mel = rand_mel(B)
    valid_frames = torch.tensor([MEL_FRAMES, AUDIO_VALID])
    torch.manual_seed(9)
    codes = torch.randint(0, TINY['n_audio_tokens'], (B, T_frames))

    def incremental(codes, codes_valid):
        """前缀 + <audio_start> + 逐帧真值输入的增量递推（复刻 generate_audio 的步进）。"""
        with torch.no_grad():
            toks, mods, masks, n_prefix = model._build_prefix_segments(
                mel, None, ids, None, valid_frames)
            x, modality_ids, key_mask = model._pack_segments(toks, mods, masks)
            states = [None] * N_LAYERS
            buffers = [None] * N_LAYERS
            positions = torch.zeros(B, 0, dtype=torch.long)
            for t in range(x.shape[1]):
                positions = torch.cat(
                    [positions, torch.full((B, 1), t, dtype=torch.long)], dim=1)
                _, states, buffers = model._forward_one_token(
                    x[:, t], modality_ids[:, t], positions, modality_ids[:, :t + 1],
                    n_prefix, states, buffers,
                    None if key_mask is None else key_mask[:, :t + 1])

            cur = model.modality_embed(
                model.audio_start_embed.view(1, 1, -1).expand(B, 1, -1).to(x.dtype),
                Modality.AUDIO)[:, 0]
            out = []
            for k in range(codes.shape[1]):
                positions = torch.cat(
                    [positions, torch.full((B, 1), positions.shape[1],
                                           dtype=torch.long)], dim=1)
                mod_t = torch.full((B,), Modality.AUDIO, dtype=torch.long)
                modality_ids = torch.cat([modality_ids, mod_t.unsqueeze(1)], dim=1)
                if key_mask is not None:
                    key_mask = torch.cat(
                        [key_mask, codes_valid[:, k].unsqueeze(1)], dim=1)
                h_t, states, buffers = model._forward_one_token(
                    cur, mod_t, positions, modality_ids, n_prefix, states, buffers,
                    key_mask)
                logits = model.audio_head(model.out_norm(h_t))
                out.append(logits)
                cur = model.modality_embed(
                    model.audio_code_embed(codes[:, k]).unsqueeze(1),
                    Modality.AUDIO)[:, 0]
                cur = torch.where(codes_valid[:, k].unsqueeze(-1), cur,
                                  model.pad_embed.to(cur.dtype).view(1, -1))
            return torch.stack(out, dim=1)

    def teacher_forced(codes, codes_valid):
        with torch.no_grad():
            return model(input_ids=ids, audio_mel=mel, audio_codes_target=codes,
                         audio_codes_valid=codes_valid,
                         audio_valid_frames=valid_frames)['audio_target_logits']

    all_valid = torch.ones(B, T_frames, dtype=torch.bool)
    tf, inc = teacher_forced(codes, all_valid), incremental(codes, all_valid)
    assert float((inc - tf).abs().max()) < 1e-5, "全有效码下增量递推 ≠ 整段前向"

    padded = all_valid.clone()
    padded[1, 5:] = False
    tf_p, inc_p = teacher_forced(codes, padded), incremental(codes, padded)
    mask = padded.unsqueeze(-1).expand_as(inc_p)
    assert float((inc_p - tf_p)[mask].abs().max()) < 1e-5, \
        "尾 padding 下有效位的增量递推 ≠ 整段前向"
    assert torch.equal(inc_p.argmax(-1)[padded], tf_p.argmax(-1)[padded])

    # 贪心生成的第 0 帧必须等于训练路径第 0 帧的 argmax
    with torch.no_grad():
        gen = model.generate_audio(input_ids=ids, audio_mel=mel,
                                   audio_valid_frames=valid_frames,
                                   n_frames=T_frames, greedy=True)
    assert tuple(gen.shape) == (B, T_frames), gen.shape
    assert int(gen.min()) >= 0 and int(gen.max()) < TINY['n_audio_tokens']
    assert torch.equal(gen[:, 0], tf[:, 0].argmax(-1)), \
        "generate_audio 第 0 帧与训练路径 argmax 不一致"

    # 生成逐样本独立：单样本生成 == batch 内对应行
    with torch.no_grad():
        gen1 = model.generate_audio(input_ids=ids[1:2], audio_mel=mel[1:2],
                                    audio_valid_frames=valid_frames[1:2],
                                    n_frames=T_frames, greedy=True)
    assert torch.equal(gen1[0], gen[1]), "生成存在跨样本污染"


# ═══════════════════════════════════════════════════════════════
# 运行器
# ═══════════════════════════════════════════════════════════════

TESTS = [
    test_state_updates_and_persists,        # §4-1
    test_bptt_through_state,                # §4-2
    test_bptt_full_model,                   # §4-2
    test_no_cross_sample_contamination,     # §4-3
    test_causality_golden_text,             # §4-4
    test_causality_golden_cross_modal,      # §4-4
    test_causal_mask_rules,                 # §4-5
    test_sparse_scores_signed,              # §4-6
    test_ersa_long_sequence,                # §4-7
    test_audio_valid_len_matches_encoder,   # B8
    test_padding_invariance_model,          # §4-9
    test_padding_invariance_loss,           # §4-9
    test_save_load_roundtrip,               # §4-10
    test_generate_audio_matches_teacher_forcing,   # §4-11（生成↔训练同源）
]


def main() -> int:
    print("=" * 64)
    print("P1 验收测试（1~7、9、10、11）")
    print("=" * 64)
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Exception as e:
            failed.append(fn.__name__)
            print(f"\n  ✗ {fn.__name__}\n{traceback.format_exc()}")
        else:
            print(f"  ✓ {fn.__name__}")
    print("-" * 64)
    print(f"通过 {len(TESTS) - len(failed)}/{len(TESTS)}")
    print("未在线覆盖：§4-8（真实 audio target 的可学性）、§4-11 后半（真实音频闭环）"
          "→ tests/overfit_audio_small.py + tests/closed_loop_audio.py"
          "（+ tests/audio_exposure_curve.py 为闭环的诊断补充）")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

