# 1. mtp.py compat shim for vLLM 0.34 (mtp_validation moved)
p = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/mtp.py"
src = open(p).read()
old = """from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)"""
new = """try:
    from vllm.model_executor.model_loader.mtp_validation import (
        is_mtp_completeness_check_enabled,
    )
except ImportError:
    def is_mtp_completeness_check_enabled(*_a, **_kw) -> bool:
        return False"""
assert old in src, "mtp.py target not found"
open(p, "w").write(src.replace(old, new, 1))

# 2. patches.py compat shim for vLLM 0.34 SpeculativeConfig.hf_config_override(*args, **kwargs)
p2 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/patches.py"
src2 = open(p2).read()
old2 = """    def hf_config_override(hf_config):
        return orig_override(_ernie5_hf_config_override(hf_config))"""
new2 = """    def hf_config_override(hf_config, *args, **kwargs):
        return orig_override(_ernie5_hf_config_override(hf_config), *args, **kwargs)"""
assert old2 in src2, "patches.py target not found"
open(p2, "w").write(src2.replace(old2, new2, 1))

# 3. config.py compat shim for vLLM 0.34 DeepseekV4Attention layer_types / compress_rates
p3 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/config.py"
src3 = open(p3).read()
old3 = """        self.compress_ratios = (
            compress_ratios
            if compress_ratios is not None
            else [128] * (num_hidden_layers + num_nextn_predict_layers)
        )"""
new3 = """        self.compress_ratios = (
            compress_ratios
            if compress_ratios is not None
            else [128] * (num_hidden_layers + num_nextn_predict_layers)
        )
        self.layer_types = [
            "heavily_compressed_attention" if r > 1 else "full_attention"
            for r in self.compress_ratios[:num_hidden_layers]
        ]
        self.compress_rates = {
            "heavily_compressed_attention": max((r for r in self.compress_ratios if r > 1), default=128)
        }"""
assert old3 in src3, "config.py target not found"
open(p3, "w").write(src3.replace(old3, new3, 1))

# 4. attention.py: SM120/SM121 (GB10 DGX Spark) native FlashInfer sparse MLA for HCA + DSA (with exact sink & 2048+128 LSE merge)
p4 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/attention.py"
src4 = open(p4).read()

# 4a. Import DeepseekV4FlashInferSM120Attention alongside DeepseekV4FlashInferMLAAttention
old4a = """from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
)"""
new4a = """from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)"""
assert old4a in src4, "attention.py import target not found"
src4 = src4.replace(old4a, new4a, 1)

# 4b. Add Ernie5_5HCAFlashInferSM120Attention and route SM12x HCA layers to it
old4b = """class Ernie5_5HCAFlashMLAAttention(
    Ernie5_5HCAAttentionMixin, DeepseekV4FlashMLAAttention
):
    \"\"\"HCA layer on FlashMLA sparse MLA (``fp8_ds_mla`` KV cache).\"\"\"


def select_ernie5_5_hca_cls(kv_cache_dtype: str) -> type[DeepseekV4Attention]:
    \"\"\"FlashInfer for the plain bf16 KV row, FlashMLA for ``fp8_ds_mla``.\"\"\"
    if kv_cache_dtype.startswith("fp8"):
        return Ernie5_5HCAFlashMLAAttention
    return Ernie5_5HCAFlashInferAttention"""

new4b = """class Ernie5_5HCAFlashMLAAttention(
    Ernie5_5HCAAttentionMixin, DeepseekV4FlashMLAAttention
):
    \"\"\"HCA layer on FlashMLA sparse MLA (``fp8_ds_mla`` KV cache).\"\"\"


class Ernie5_5HCAFlashInferSM120Attention(
    Ernie5_5HCAAttentionMixin, DeepseekV4FlashInferSM120Attention
):
    \"\"\"HCA layer on FlashInfer SM120 sparse MLA (``fp8_ds_mla`` KV cache, GB10/SM12x).\"\"\"


def select_ernie5_5_hca_cls(kv_cache_dtype: str) -> type[DeepseekV4Attention]:
    \"\"\"FlashInfer SM120 on SM12x, FlashMLA for fp8 on SM90/100, FlashInfer for bf16.\"\"\"
    cap = current_platform.get_device_capability()
    if cap is not None and cap.major == 12:
        return Ernie5_5HCAFlashInferSM120Attention
    if kv_cache_dtype.startswith("fp8"):
        return Ernie5_5HCAFlashMLAAttention
    return Ernie5_5HCAFlashInferAttention"""
assert old4b in src4, "attention.py select_ernie5_5_hca_cls target not found"
src4 = src4.replace(old4b, new4b, 1)

# 4c. Add Ernie5_5FlashInferMLASparseSM120Impl & Backend with exact sink + 2048+128 LSE merge
old4c = """class Ernie5_5FlashMLASparseBackend(FlashMLASparseBackend):
    # Keeps the parent name: shared code keys the fp8_ds_mla KV layout on
    # "FLASHMLA_SPARSE".

    get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

    @staticmethod
    def get_impl_cls() -> type[Ernie5_5FlashMLASparseImpl]:
        return Ernie5_5FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True"""

new4c = old4c + """


_SM120_LN2 = 0.6931471805599453


def _make_sm120_dsa_backend():
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend,
        _finish_dcp_partials,
        _get_workspace_buffer,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )
    from vllm.utils.flashinfer import (
        flashinfer_trtllm_batch_decode_with_kv_cache_mla,
    )

    class Ernie5_5FlashInferMLASparseSM120Impl(FlashInferMLASparseSM120Impl):
        def __init__(self, *args, **kwargs):
            self.sinks: torch.Tensor = kwargs.pop("sinks")
            super().__init__(*args, **kwargs)

        def _call_fi_slice(
            self,
            q: torch.Tensor,
            kv_cache: torch.Tensor,
            indices: torch.Tensor,
            valid_counts: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            num_toks = q.shape[0]
            cap = indices.shape[1]
            out = q.new_empty((num_toks, q.shape[1], self.kv_lora_rank), dtype=q.dtype)
            lse = q.new_empty((num_toks, q.shape[1]), dtype=torch.float32)
            flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1),
                kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
                workspace_buffer=_get_workspace_buffer(q.device),
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=indices.unsqueeze(1),
                seq_lens=valid_counts,
                max_seq_len=cap,
                out=out.unsqueeze(1),
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=cap,
                kv_scale_format=self.kv_scale_format,
                return_lse=True,
                lse=lse,
            )
            if valid_counts is not None:
                _finish_dcp_partials(out, lse, valid_counts)
            return out, lse

        def _run_sparse_kernel(
            self,
            q: torch.Tensor,
            kv_cache: torch.Tensor,
            topk_indices_physical: torch.Tensor,
            valid_counts: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            cap = topk_indices_physical.shape[1]
            sinks = self.sinks.to(device=q.device, dtype=torch.float32).unsqueeze(0)
            if cap > 2048:
                idx1 = topk_indices_physical[:, :2048].contiguous()
                idx2 = topk_indices_physical[:, 2048:].contiguous()
                out1, lse1 = self._call_fi_slice(q, kv_cache, idx1, valid_counts)
                out2, lse2 = self._call_fi_slice(q, kv_cache, idx2, valid_counts)
                l1 = lse1 * _SM120_LN2
                l2 = lse2 * _SM120_LN2
                ls = sinks.expand_as(l1)
                m = torch.maximum(torch.maximum(l1, l2), ls)
                f1 = torch.isfinite(l1)
                f2 = torch.isfinite(l2)
                w1 = torch.where(f1, torch.exp(torch.where(f1, l1 - m, 0.0)), 0.0)
                w2 = torch.where(f2, torch.exp(torch.where(f2, l2 - m, 0.0)), 0.0)
                ws = torch.exp(ls - m)
                denom = w1 + w2 + ws
                o1 = torch.where(f1.unsqueeze(-1), out1.float(), 0.0)
                o2 = torch.where(f2.unsqueeze(-1), out2.float(), 0.0)
                out = (
                    o1 * (w1 / denom).unsqueeze(-1)
                    + o2 * (w2 / denom).unsqueeze(-1)
                ).to(out1.dtype)
                lse_out = (
                    torch.log2(denom) + (m / _SM120_LN2)
                    if self.need_to_return_lse_for_decode
                    else None
                )
                return out, lse_out
            out1, lse1 = self._call_fi_slice(
                q, kv_cache, topk_indices_physical, valid_counts
            )
            l1 = lse1 * _SM120_LN2
            f1 = torch.isfinite(l1)
            scale = torch.where(f1, torch.sigmoid(torch.where(f1, l1 - sinks, 0.0)), 0.0)
            out = torch.where(
                f1.unsqueeze(-1), out1.float() * scale.unsqueeze(-1), 0.0
            ).to(out1.dtype)
            return out, (lse1 if self.need_to_return_lse_for_decode else None)

    class Ernie5_5FlashInferMLASparseSM120Backend(FlashInferMLASparseSM120Backend):
        get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

        @staticmethod
        def get_impl_cls():
            return Ernie5_5FlashInferMLASparseSM120Impl

        @classmethod
        def supports_sink(cls) -> bool:
            return True

    return Ernie5_5FlashInferMLASparseSM120Backend


def _select_dsa_backend():
    cap = current_platform.get_device_capability()
    if cap is not None and cap.major == 12:
        return _make_sm120_dsa_backend()
    return Ernie5_5FlashMLASparseBackend"""
assert old4c in src4, "attention.py Ernie5_5FlashMLASparseBackend target not found"
src4 = src4.replace(old4c, new4c, 1)

# 4d. Use _select_dsa_backend() in _check_sink_backend and Ernie5_5DSAAttention
old4d = "    invalid_reasons = Ernie5_5FlashMLASparseBackend.validate_configuration("
new4d = "    invalid_reasons = _select_dsa_backend().validate_configuration("
assert old4d in src4, "attention.py _check_sink_backend target not found"
src4 = src4.replace(old4d, new4d, 1)

old4e = "            attn_backend=Ernie5_5FlashMLASparseBackend,"
new4e = "            attn_backend=_select_dsa_backend(),"
assert old4e in src4, "attention.py Ernie5_5DSAAttention attn_backend target not found"
src4 = src4.replace(old4e, new4e, 1)

open(p4, "w").write(src4)
print("Patched mtp.py, patches.py, config.py, and attention.py (SM120/121 HCA+DSA) cleanly!")

# 5. model.py: vLLM 0.34 replaced spec_config.use_eagle()/uses_draft_model() with needs_mtp_hidden_buffer
p5 = f"/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/model.py"
src5 = open(p5).read()
old5 = """        if (
            get_pp_group().is_last_rank
            and spec_config is not None
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
        ):"""
new5 = """        from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer
        if get_pp_group().is_last_rank and needs_mtp_hidden_buffer(spec_config):"""
assert old5 in src5, "model.py spec_config target not found"
src5 = src5.replace(old5, new5, 1)
open(p5, "w").write(src5)
print("Patched model.py (needs_mtp_hidden_buffer) cleanly!")

# 6. attention.py: vLLM 0.34 renamed DeepseekV4Attention._prepare_and_attn_fn -> _prepare_and_attn
p6 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/attention.py"
src6 = open(p6).read()
assert "self._prepare_and_attn_fn(" in src6, "attention.py _prepare_and_attn_fn target not found"
src6 = src6.replace("self._prepare_and_attn_fn(", "self._prepare_and_attn(", 1)
open(p6, "w").write(src6)
print("Patched attention.py (_prepare_and_attn) cleanly!")

# 7. attention.py: vLLM 0.34 renamed get_kv_cache_spec -> get_kv_cache_config (returns KVCacheLayerConfig)
p7 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/attention.py"
src7 = open(p7).read()
old7 = """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _dsa_page_spec(super().get_kv_cache_spec(vllm_config), vllm_config)"""
new7 = """    def get_kv_cache_config(self, vllm_config: VllmConfig):
        cfg = super().get_kv_cache_config(vllm_config)
        return dataclasses.replace(cfg, spec=_dsa_page_spec(cfg.spec, vllm_config))"""
assert src7.count(old7) == 2, f"Expected 2 get_kv_cache_spec overrides, got {src7.count(old7)}"
src7 = src7.replace(old7, new7)
open(p7, "w").write(src7)
print("Patched attention.py (get_kv_cache_config -> DSA block_size=64) cleanly!")

# 8. b12x situ.py: forward all MoEDynamicKernelBackend kwargs (w4a8_n64_repacked, etc.) cleanly
import glob
patched_situ = 0
for p8 in glob.glob("/usr/local/lib/python3.12/dist-packages/**/b12x/moe/_shared/kernels/situ.py", recursive=True):
    src8 = open(p8).read()
    idx = src8.find("class MoEDynamicKernelSitu(MoEDynamicKernelBackend):")
    assert idx != -1, f"MoEDynamicKernelSitu not found in {p8}"
    end = src8.find("__all__ =", idx)
    assert end != -1, f"__all__ not found in {p8}"
    replacement = """class MoEDynamicKernelSitu(MoEDynamicKernelBackend):
    def __init__(self, *args: object, **kwargs: object):
        kwargs["activation"] = SITU
        super().__init__(*args, **kwargs)


"""
    src8 = src8[:idx] + replacement + src8[end:]
    open(p8, "w").write(src8)
    patched_situ += 1
assert patched_situ > 0, "No b12x situ.py found to patch"
print(f"Patched {patched_situ} b12x situ.py file(s) cleanly!")

# 9. flashindexer.py: vLLM 0.34 passes extra kwargs (e.g. logits_dtype) to fp8_fp4_mqa_logits
p9 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/nvidia/flashindexer.py"
src9 = open(p9).read()
old9a = """def _flashindexer_fp8_fp4_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
):"""
new9a = """def _flashindexer_fp8_fp4_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True, **kwargs
):"""
assert old9a in src9, "flashindexer.py def target not found"
src9 = src9.replace(old9a, new9a, 1)
src9 = src9.replace("clean_logits=clean_logits\n        )", "clean_logits=clean_logits,\n            **kwargs,\n        )")
src9 = src9.replace("clean_logits=clean_logits,\n    )", "clean_logits=clean_logits,\n        **kwargs,\n    )")
open(p9, "w").write(src9)
print("Patched flashindexer.py (**kwargs forwarding) cleanly!")

# 9b. vLLM 0.34 made the 6th positional arg max_seqlen_k (int) and dropped clean_logits; the plugin wrapper
# still forwarded it as clean_logits=..., so enabling FlashIndexer died at warm-up. Forward positionals instead.
src9 = open(p9).read()
_n = src9.count("clean_logits=clean_logits,")
src9 = src9.replace("cu_seqlen_ke, clean_logits=True, **kwargs\n):", "cu_seqlen_ke, *args, **kwargs\n):", 1)
src9 = src9.replace("clean_logits=clean_logits,", "*args,")
_body = src9.split("def _flashindexer_fp8_fp4_mqa_logits")[1].split("def patch_sparse")[0]
assert _n == 2 and "clean_logits" not in _body and "*args, **kwargs" in _body, "9b targets"
open(p9, "w").write(src9)
print("Patched flashindexer.py (vLLM 0.34 positional max_seqlen_k) cleanly!")

# 9c. GB10's DeepGEMM MQA-logits kernel only takes 16/32/64 heads (csrc/apis/attention.hpp:142); FlashIndexer keeps 8.
# Pad the 8 selected heads to 16 by repeating them with weight exactly 0: relu(q.k) * 0 adds nothing, so the logits are
# the 8-head selection's, computed by a 16-head kernel (4x fewer heads than the full 64 instead of 8x).
src9 = open(p9).read()
old9c = "    return _orig_mqa_logits(\n        (q_sel, q_sf_sel),"
new9c = """    if q_sel.shape[1] < 16:
        _pad = 16 - q_sel.shape[1]
        _qb = q_sel.contiguous().view(torch.uint8)
        q_sel = torch.cat([_qb, _qb[:, :_pad]], dim=1).view(q_sel.dtype)
        if q_sf_sel is not None:
            q_sf_sel = torch.cat([q_sf_sel, q_sf_sel[:, :_pad]], dim=1)
        w_sel = torch.cat([w_sel, torch.zeros_like(w_sel[:, :_pad])], dim=1)
""" + old9c
assert src9.count(old9c) == 1, "9c target"
src9 = src9.replace(old9c, new9c, 1)
open(p9, "w").write(src9)
print("Patched flashindexer.py (GB10 16-head pad) cleanly!")



# 10. parser.py: disable minimax_m2 arg_value_extender (vLLM 0.34 optimization conflicts with
# _ernie5_arg_converter popping empty trailing params, corrupting streamed multi-arg JSON tool calls)
p10 = "/usr/local/lib/python3.12/dist-packages/vllm_ernie/models/ernie5_5/parser.py"
src10 = open(p10).read()
old10 = "        arg_converter=_ernie5_arg_converter,\n    )"
new10 = "        arg_converter=_ernie5_arg_converter,\n        arg_value_extender=None,\n    )"
assert old10 in src10, "parser.py ernie5_config target not found"
src10 = src10.replace(old10, new10, 1)
open(p10, "w").write(src10)
print("Patched parser.py (arg_value_extender=None for valid streaming multi-arg tool calls) cleanly!")

# 11. GB10 (sm_12x) low-M bf16 x bf16 -> fp32 GEMVs through vLLM's cuteDSL ll_bf16_gemm instead of cuBLAS / fp32 F.linear.
# cuBLAS drops to an SM80 16x16 WMMA kernel for narrow N at 2-16 tokens on sm_12x (vLLM's own note in layers/utils.py), and
# GateLinear's specialized router tiers are gated to Hopper / Blackwell-100. Both GEMVs below are bf16 x bf16 with fp32
# accumulation and fp32 output before and after; only the summation order changes. Applied for M <= 16 only.
import os as _os11
_P11 = "/usr/local/lib/python3.12/dist-packages"

# 11a. DeepSeek-V4 compressor fused wkv/wgate projection (ERNIE's HCA layers): torch.mm(out_dtype=fp32) -> ll_bf16_gemm.
p11a = _P11 + "/vllm/models/deepseek_v4/compressor.py"
s11a = open(p11a).read()
old11a = """class DeepseekV4CompressorKVScoreLinear(MergedColumnParallelLinear):
    \"\"\"Fused `wkv`/`wgate` projection with the FP32 output the compressor
    state accumulates in.\"\"\"

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.mm(input_, self.weight.T, out_dtype=torch.float32)"""
new11a = """def _sm12x_ll_bf16_gemm():
    try:
        if not (current_platform.is_cuda() and current_platform.is_device_capability_family(120)):
            return None
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import is_available, ll_bf16_gemm
        return ll_bf16_gemm if is_available() else None
    except Exception:
        return None


_SM12X_LL_BF16_GEMM = _sm12x_ll_bf16_gemm()


class DeepseekV4CompressorKVScoreLinear(MergedColumnParallelLinear):
    \"\"\"Fused `wkv`/`wgate` projection with the FP32 output the compressor
    state accumulates in.\"\"\"

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if (
            _SM12X_LL_BF16_GEMM is not None
            and input_.dim() == 2
            and input_.shape[0] <= 16
            and input_.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            return _SM12X_LL_BF16_GEMM(input_, self.weight, torch.float32)
        return torch.mm(input_, self.weight.T, out_dtype=torch.float32)"""
assert s11a.count(old11a) == 1, "compressor.py DeepseekV4CompressorKVScoreLinear target not found"
open(p11a, "w").write(s11a.replace(old11a, new11a, 1))
print("Patched compressor.py (sm_12x ll_bf16_gemm for M<=16) cleanly!")

# 11b. ERNIE router: keep the gate in the checkpoint's bf16 and let GateLinear's tier 1 (ll_bf16_gemm, fp32 out) run on sm_12x.
p11b = _P11 + "/vllm_ernie/models/ernie5_5/nvidia/moe.py"
s11b = open(p11b).read()
old11b = """        self.gate = GateLinear(
            hidden_size,
            2 * num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )"""
new11b = """        _sm12x_ll = _ernie_sm12x_ll_bf16_available()
        self.gate = GateLinear(
            hidden_size,
            2 * num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.bfloat16 if _sm12x_ll else torch.float32,
            prefix=f"{prefix}.gate",
        )
        if _sm12x_ll:
            self.gate.allow_ll_bf16_gemm = True"""
assert s11b.count(old11b) == 1, "moe.py Ernie5_5MoE gate target not found"
s11b = s11b.replace(old11b, new11b, 1)
old11c = "class Ernie5_5MoE(nn.Module):"
new11c = """def _ernie_sm12x_ll_bf16_available() -> bool:
    try:
        from vllm.platforms import current_platform

        if not (current_platform.is_cuda() and current_platform.is_device_capability_family(120)):
            return False
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import is_available

        return bool(is_available())
    except Exception:
        return False


class Ernie5_5MoE(nn.Module):"""
assert s11b.count(old11c) == 1, "moe.py Ernie5_5MoE class target not found"
open(p11b, "w").write(s11b.replace(old11c, new11c, 1))
print("Patched moe.py (bf16 router weights + GateLinear ll_bf16_gemm tier on sm_12x) cleanly!")
