import argparse
import logging as pylogging
import math
import os
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_m, logits_for_images, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validator gate constants (mirror perturbnet/constants.py defaults).
# ---------------------------------------------------------------------------
VALIDATOR_MIN_LINF = 0.003
VALIDATOR_MAX_LINF = 0.03
VALIDATOR_MIN_SSIM = 0.98
VALIDATOR_MIN_PSNR_DB = 38.0

# One uint8 quantization step. PNG round-trip can shift any pixel by up to
# this amount, so we keep a safety margin away from every gate threshold.
QUANT_STEP = 1.0 / 255.0
SAFE_MIN_LINF = VALIDATOR_MIN_LINF + QUANT_STEP    # ~0.00692
SAFE_MAX_LINF = VALIDATOR_MAX_LINF - QUANT_STEP   # ~0.02608
TARGET_SSIM = 0.985
TARGET_PSNR_DB = 39.0


def _quantize_snap(image_chw: torch.Tensor) -> torch.Tensor:
    """Snap to the exact uint8 grid the PNG round-trip will produce."""
    return (image_chw.clamp(0.0, 1.0) * 255.0).round() / 255.0


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    a = x_clean.unsqueeze(0).float()
    b = x_adv.unsqueeze(0).float()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = F.avg_pool2d(a, kernel_size, 1, padding)
    mu_b = F.avg_pool2d(b, kernel_size, 1, padding)
    sa = F.avg_pool2d(a * a, kernel_size, 1, padding) - mu_a * mu_a
    sb = F.avg_pool2d(b * b, kernel_size, 1, padding) - mu_b * mu_b
    sab = F.avg_pool2d(a * b, kernel_size, 1, padding) - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sa + sb + c2)
    return float((num / (den + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config):
    resolved_config = config() if callable(config) else config
    axon_config = getattr(resolved_config, "axon", None)
    axon_kwargs = {"wallet": wallet}
    if axon_config is not None:
        for key in ("port", "ip", "external_port", "external_ip", "max_workers"):
            value = getattr(axon_config, key, None)
            if value is not None:
                axon_kwargs[key] = value

    if hasattr(bt, "axon"):
        try:
            return bt.axon(**axon_kwargs)
        except TypeError:
            try:
                return bt.axon(wallet=wallet, config=resolved_config)
            except Exception as exc:
                raise RuntimeError(f"Failed to build axon with kwargs/config: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to build axon with kwargs: {exc}") from exc
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    try:
        return axon_cls(**axon_kwargs)
    except TypeError:
        try:
            return axon_cls(wallet=wallet, config=resolved_config)
        except Exception as exc:
            raise RuntimeError(f"Failed to build Axon class with kwargs/config: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to build Axon class with kwargs: {exc}") from exc


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Force deterministic cuDNN so search-time predictions match round-trip
        # predictions exactly (avoids borderline candidates flipping back).
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        self.model = load_efficientnet_v2_m(self.device)
        self._prewarm_model()

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _prewarm_model(self) -> None:
        """Run a single forward pass so kernels and the autotuner are ready."""
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 128, 128, device=self.device)
                _ = logits_for_images(model=self.model, image_bchw=dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("[MINER] Model pre-warmed.")
        except Exception as exc:
            logger.warning(f"[MINER] Model pre-warm failed (continuing): {exc}")

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------
    def _verify_candidate(
        self,
        clean: torch.Tensor,
        cand: torch.Tensor,
        true_idx: int,
        d_eff: float,
        min_n: float,
        margin: float = 0.0,
    ) -> typing.Tuple[bool, str, dict]:
        """Apply every validator gate, with safety margins, on a quantized cand.

        ``margin`` requires the predicted class's logit to exceed the true
        class's logit by at least this amount; protects against borderline
        candidates that round-trip flips back to the true label.
        """
        if cand.shape != clean.shape:
            return False, "shape_mismatch", {}
        if cand.min().item() < 0.0 or cand.max().item() > 1.0:
            return False, "value_out_of_range", {}
        norm = float((cand - clean).abs().max().item())
        if norm < min_n:
            return False, "below_min_delta", {"norm": norm}
        if norm > d_eff:
            return False, "above_max_delta", {"norm": norm}
        with torch.no_grad():
            logits = logits_for_images(
                model=self.model, image_bchw=cand.unsqueeze(0)
            )[0]
        pred = int(logits.argmax().item())
        if pred == true_idx:
            return False, "label_match_with_original", {"norm": norm, "pred": pred}
        if margin > 0.0:
            gap = float((logits[pred] - logits[true_idx]).item())
            if gap < margin:
                return False, "below_margin", {"norm": norm, "pred": pred, "gap": gap}
        ssim = _compute_ssim(clean, cand)
        if ssim < TARGET_SSIM:
            return False, "below_target_ssim", {"norm": norm, "ssim": ssim, "pred": pred}
        psnr = _compute_psnr_db(clean, cand)
        if psnr < TARGET_PSNR_DB:
            return False, "below_target_psnr", {"norm": norm, "ssim": ssim, "psnr": psnr, "pred": pred}
        return True, "ok", {"norm": norm, "ssim": ssim, "psnr": psnr, "pred": pred}

    # ------------------------------------------------------------------
    # Stage 1: Sparse JSMA-style attack — perturb the top-K saliency
    # (pixel, channel) entries by EXACTLY one quantization step (1/255).
    # Includes:
    #   * per-channel saliency (3× cheaper RMSE budget vs per-pixel)
    #   * top-T target classes, pick the one needing fewest mods
    #   * batched single-removal greedy pruning after binary search
    # ------------------------------------------------------------------
    def _sparse_jsma_attack(
        self,
        clean: torch.Tensor,
        true_idx: int,
        d_eff: float,
        min_n: float,
        top_targets: int = 10,
        max_mods: int = 4096,
        prune_chunk: int = 256,
        good_enough_k: int = 32,
        prune_min_k: int = 8,
        moderate_k: int = 96,
        prune_passes: int = 2,
        margin: float = 0.0,
        step_mul: int = 1,
    ) -> typing.Optional[typing.Tuple[torch.Tensor, int, int]]:
        """Returns (cand, target_idx, k_final) or None.

        ``step_mul`` perturbs each selected (pixel, channel) by ±step_mul/255
        instead of ±1/255. step_mul=2 lets us flip "harder" images while still
        staying at L∞ = 2/255 — far cheaper than dense DeepFool fallback.

        ``prune_chunk`` is the GPU batch size for the single-removal trial;
        the pruner now works at any K by chunking, instead of being skipped
        for large K. ``prune_passes`` runs the pruner repeatedly until it
        stops shrinking, since removing one pixel can unlock removing more.
        """
        try:
            C, H, W = clean.shape
            step = QUANT_STEP * float(step_mul)

            x = clean.unsqueeze(0).detach().requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=x)[0]
            masked = logits.detach().clone()
            masked[true_idx] = float("-inf")
            t_top = min(top_targets, masked.numel() - 1)
            target_list: typing.List[int] = torch.topk(masked, k=t_top).indices.tolist()

            best_overall: typing.Optional[typing.Tuple[torch.Tensor, int, int]] = None

            for t_pos, target_idx in enumerate(target_list):
                target_margin = logits[target_idx] - logits[true_idx]
                grad = torch.autograd.grad(
                    target_margin, x, retain_graph=(t_pos < len(target_list) - 1)
                )[0][0]  # (C,H,W)

                # Per (pixel, channel) saliency — flatten across channels too.
                sal_flat = grad.abs().flatten()                         # (N,)
                order = torch.argsort(sal_flat, descending=True)         # (N,)
                sign_delta_flat = (grad.sign() * step).flatten()         # (N,)

                def _build_cand(idx_tensor: torch.Tensor) -> torch.Tensor:
                    delta_flat = torch.zeros_like(sign_delta_flat)
                    delta_flat[idx_tensor] = sign_delta_flat[idx_tensor]
                    return _quantize_snap(clean + delta_flat.view(C, H, W))

                def _try_k(k: int) -> typing.Optional[torch.Tensor]:
                    cand = _build_cand(order[:k])
                    ok, _r, _info = self._verify_candidate(
                        clean, cand, true_idx, d_eff, min_n, margin=margin
                    )
                    return cand if ok else None

                # 1) Geometric escalation until the flip succeeds.
                schedule = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                best_cand: typing.Optional[torch.Tensor] = None
                best_k = -1
                for k in schedule:
                    if k > max_mods:
                        break
                    c = _try_k(k)
                    if c is not None:
                        best_cand = c
                        best_k = k
                        break
                if best_cand is None:
                    continue

                # 2) Binary-search downward for smallest contiguous prefix.
                lo, hi = max(4, best_k // 4), best_k
                while lo < hi:
                    mid = (lo + hi) // 2
                    c = _try_k(mid)
                    if c is not None:
                        best_cand = c
                        best_k = mid
                        hi = mid
                    else:
                        lo = mid + 1

                # 3) Greedy pruning via chunked single-removal trial. Runs
                #    multiple passes since removing one pixel can unlock more.
                kept = order[:best_k].clone()
                for _pass in range(prune_passes):
                    cur_k = int(kept.numel())
                    if cur_k < prune_min_k:
                        break
                    try:
                        with torch.no_grad():
                            base_delta = torch.zeros_like(sign_delta_flat)
                            base_delta[kept] = sign_delta_flat[kept]
                            base_full = _quantize_snap(
                                clean + base_delta.view(C, H, W)
                            )
                            removable_chunks: typing.List[torch.Tensor] = []
                            chunk = max(1, min(prune_chunk, cur_k))
                            for start in range(0, cur_k, chunk):
                                end = min(start + chunk, cur_k)
                                bs = end - start
                                trials = base_full.unsqueeze(0).repeat(bs, 1, 1, 1)
                                for i in range(bs):
                                    idx = int(kept[start + i].item())
                                    ch = idx // (H * W)
                                    rem = idx - ch * (H * W)
                                    hh = rem // W
                                    ww = rem - hh * W
                                    trials[i, ch, hh, ww] = clean[ch, hh, ww]
                                preds = logits_for_images(
                                    model=self.model, image_bchw=trials
                                ).argmax(dim=1)
                                rem_local = (preds != true_idx).nonzero(as_tuple=True)[0]
                                if rem_local.numel() > 0:
                                    removable_chunks.append(rem_local + start)
                        if not removable_chunks:
                            break
                        removable = torch.cat(removable_chunks)
                        keep_mask = torch.ones(
                            cur_k, dtype=torch.bool, device=clean.device
                        )
                        keep_mask[removable] = False
                        new_kept = kept[keep_mask]
                        if new_kept.numel() == 0:
                            break
                        cand_new = _build_cand(new_kept)
                        ok, _r, _info = self._verify_candidate(
                            clean, cand_new, true_idx, d_eff, min_n, margin=margin
                        )
                        if not ok:
                            break
                        best_cand = cand_new
                        best_k = int(new_kept.numel())
                        kept = new_kept
                    except Exception as exc:
                        logger.warning(f"[sparse_jsma] prune failed: {exc}")
                        break

                if best_overall is None or best_k < best_overall[2]:
                    best_overall = (best_cand, target_idx, best_k)
                # Early-exit tier 1: very small mod set — done.
                if best_overall is not None and best_overall[2] <= good_enough_k:
                    break
                # Early-exit tier 2: only spend extra latency on more targets
                # if the first target's K was clearly mediocre.
                if t_pos == 0 and best_overall is not None and best_overall[2] <= moderate_k:
                    break

            return best_overall
        except Exception as exc:
            logger.warning(f"[sparse_jsma] failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Stage 2: DeepFool-Linf multi-target probe (1 batched forward + k+1 backwards)
    # ------------------------------------------------------------------
    def _deepfool_linf_multi(
        self, clean: torch.Tensor, true_idx: int, top_k: int = 5
    ) -> typing.Optional[typing.Tuple[torch.Tensor, int, float]]:
        try:
            x = clean.unsqueeze(0).detach().requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=x)[0]
            masked = logits.detach().clone()
            masked[true_idx] = float("-inf")
            top = torch.topk(masked, k=min(top_k, masked.numel() - 1)).indices.tolist()

            grad_true = torch.autograd.grad(logits[true_idx], x, retain_graph=True)[0][0]
            best_delta: typing.Optional[torch.Tensor] = None
            best_target = -1
            best_norm = float("inf")
            for j in top:
                grad_j = torch.autograd.grad(logits[j], x, retain_graph=True)[0][0]
                w = (grad_j - grad_true).detach()                       # (3,H,W)
                f_diff = float((logits[j] - logits[true_idx]).item())   # < 0 (true wins)
                if f_diff >= 0:
                    continue
                denom = float(w.abs().sum().item()) + 1e-12
                r_mag = -f_diff / denom
                delta = r_mag * w.sign()
                norm = float(delta.abs().max().item())
                if norm < best_norm:
                    best_norm = norm
                    best_delta = delta
                    best_target = j
            if best_delta is None:
                return None
            return best_delta, best_target, best_norm
        except Exception as exc:
            logger.warning(f"[deepfool] failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Saliency mask: top-N% pixels by |∇logit_true| (one backward).
    # ------------------------------------------------------------------
    def _saliency_mask(self, clean: torch.Tensor, true_idx: int, top_pct: float = 8.0) -> torch.Tensor:
        try:
            x = clean.unsqueeze(0).detach().requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=x)[0]
            grad = torch.autograd.grad(logits[true_idx], x)[0][0].abs().sum(dim=0)
            flat = grad.flatten()
            k = max(1, int(flat.numel() * top_pct / 100.0))
            thresh = torch.topk(flat, k=k).values.min()
            mask2d = (grad >= thresh).float()
            return mask2d.unsqueeze(0)  # (1,H,W) broadcasts to (3,H,W)
        except Exception as exc:
            logger.warning(f"[saliency] failed ({exc}); falling back to full mask")
            return torch.ones_like(clean[:1])

    # ------------------------------------------------------------------
    # Stage 3: targeted MI-PGD with TV regularizer + saliency mask, decaying step,
    # quantization-aware acceptance.
    # ------------------------------------------------------------------
    def _mi_pgd_targeted(
        self,
        clean: torch.Tensor,
        true_idx: int,
        target_idx: int,
        init_delta: torch.Tensor,
        mask: torch.Tensor,
        radius: float,
        d_eff: float,
        min_n: float,
        steps: int = 8,
        momentum: float = 0.9,
        lambda_tv: float = 0.02,
    ) -> typing.Optional[torch.Tensor]:
        radius = max(min_n, min(radius, d_eff))
        delta = (init_delta * mask).clamp(-radius, radius).detach()
        velocity = torch.zeros_like(delta)
        step = max(radius / 4.0, 2.0 * QUANT_STEP)
        best_quant: typing.Optional[torch.Tensor] = None
        best_norm = float("inf")

        for _ in range(steps):
            adv = (clean + delta).clamp(0.0, 1.0).detach().requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=adv.unsqueeze(0))[0]
            margin = logits[true_idx] - logits[target_idx]
            tv = (
                (adv[:, 1:, :] - adv[:, :-1, :]).abs().mean()
                + (adv[:, :, 1:] - adv[:, :, :-1]).abs().mean()
            )
            loss = margin + lambda_tv * tv
            grad = torch.autograd.grad(loss, adv)[0]
            g_norm = grad / (grad.abs().mean() + 1e-12)
            velocity = momentum * velocity + g_norm

            # Step downhill on margin (we want margin negative → target wins).
            delta = (delta - step * velocity.sign() * mask).clamp(-radius, radius)
            delta = ((clean + delta).clamp(0.0, 1.0) - clean).detach()

            cand = _quantize_snap(clean + delta)
            ok, _reason, info = self._verify_candidate(clean, cand, true_idx, d_eff, min_n, margin=0.1)
            if ok:
                # Decay the step so subsequent iterations refine to a smaller delta.
                step *= 0.6
                if info["norm"] < best_norm:
                    best_norm = info["norm"]
                    best_quant = cand
        return best_quant

    # ------------------------------------------------------------------
    # Main forward: Stage 0 gates → Stage 2 DeepFool → Stage 3 MI-PGD ladder.
    # ------------------------------------------------------------------
    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        t_start = time.perf_counter()
        task_id = getattr(synapse, "task_id", "unknown")
        self._log_step_start(
            "miner_forward",
            task_id=task_id,
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )

        # ---- Stage 0: gates ----
        if synapse.norm_type != "Linf":
            logger.info(f"task={task_id}: unsupported norm_type={synapse.norm_type}; returning clean")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        true_idx = resolve_target_index(synapse.true_label)
        if true_idx is None:
            logger.warning(f"task={task_id}: unresolved true_label={getattr(synapse,'true_label',None)}; returning clean")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        try:
            clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        except Exception as exc:
            logger.warning(f"task={task_id}: decode failed ({exc}); returning clean")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        synapse_min_delta = float(getattr(synapse, "min_delta", VALIDATOR_MIN_LINF))
        d_eff = max(SAFE_MIN_LINF + QUANT_STEP, min(epsilon, VALIDATOR_MAX_LINF) - QUANT_STEP)
        min_n = max(SAFE_MIN_LINF, synapse_min_delta + QUANT_STEP)
        # Sparse stage uses exactly 1/255 step on already-on-grid pixels, so it
        # does not need the dense-attack quantization buffer; just stay safely
        # above the validator's hard 0.003 floor.
        min_n_sparse = max(VALIDATOR_MIN_LINF + 0.0005, synapse_min_delta + 0.0005)

        chosen: typing.Optional[torch.Tensor] = None
        chosen_stage = "none"
        target_idx: typing.Optional[int] = None
        chosen_info: dict = {}
        chosen_min_n = min_n  # which threshold the round-trip verify should use
        encoded_final: typing.Optional[str] = None

        def _try_finalize(cand: torch.Tensor, min_n_check: float) -> typing.Optional[
            typing.Tuple[str, dict]
        ]:
            """PNG round-trip + re-verify; returns (encoded, info) on success."""
            try:
                enc = encode_image_b64(cand)
                rt = decode_image_b64(enc).to(self.device)
                ok2, _r2, info2 = self._verify_candidate(
                    clean, rt, true_idx, d_eff, min_n_check
                )
                if ok2:
                    return enc, info2
            except Exception as exc:
                logger.warning(f"task={task_id}: finalize raised ({exc})")
            return None

        # ---- Stage 1a: Sparse JSMA, ±1/255 step (tiny RMSE) ----
        sparse = self._sparse_jsma_attack(
            clean=clean, true_idx=true_idx, d_eff=d_eff, min_n=min_n_sparse,
            margin=0.0, step_mul=1,
        )
        if sparse is not None:
            cand_sp, t_sp, k_sp = sparse
            ok, _reason, info = self._verify_candidate(
                clean, cand_sp, true_idx, d_eff, min_n_sparse, margin=0.0
            )
            if ok:
                fin = _try_finalize(cand_sp, min_n_sparse)
                if fin is not None:
                    encoded_final, chosen_info = fin
                    chosen = cand_sp
                    chosen_stage = f"sparse_jsma_k{k_sp}"
                    target_idx = t_sp
                    chosen_min_n = min_n_sparse

        # ---- Stage 1b: Sparse JSMA, ±2/255 step (still way under deepfool RMSE) ----
        if chosen is None:
            sparse2 = self._sparse_jsma_attack(
                clean=clean, true_idx=true_idx, d_eff=d_eff, min_n=min_n_sparse,
                margin=0.0, step_mul=2,
            )
            if sparse2 is not None:
                cand_sp, t_sp, k_sp = sparse2
                ok, _reason, info = self._verify_candidate(
                    clean, cand_sp, true_idx, d_eff, min_n_sparse, margin=0.0
                )
                if ok:
                    fin = _try_finalize(cand_sp, min_n_sparse)
                    if fin is not None:
                        encoded_final, chosen_info = fin
                        chosen = cand_sp
                        chosen_stage = f"sparse_jsma2_k{k_sp}"
                        target_idx = t_sp
                        chosen_min_n = min_n_sparse

        # ---- Stage 2: DeepFool-Linf multi-target ----
        delta_init = torch.zeros_like(clean)
        df = None
        if chosen is None:
            df = self._deepfool_linf_multi(clean=clean, true_idx=true_idx, top_k=5)
            if df is not None:
                delta_df, target_idx, raw_norm = df
                if raw_norm > 1e-9:
                    target_n = min(d_eff, max(min_n * 1.2, raw_norm * 1.1))
                    delta_df = delta_df * (target_n / raw_norm)
                cand = _quantize_snap(clean + delta_df)
                ok, _reason, info = self._verify_candidate(clean, cand, true_idx, d_eff, min_n, margin=0.1)
                if ok:
                    fin = _try_finalize(cand, min_n)
                    if fin is not None:
                        encoded_final, chosen_info = fin
                        chosen = cand
                        chosen_stage = "deepfool"
                        chosen_min_n = min_n
                delta_init = delta_df.detach()

        # ---- Stage 3: MI-PGD ladder, warm-started from DeepFool ----
        if chosen is None:
            if target_idx is None:
                with torch.no_grad():
                    lg = logits_for_images(model=self.model, image_bchw=clean.unsqueeze(0))[0]
                    lg = lg.clone()
                    lg[true_idx] = float("-inf")
                    target_idx = int(lg.argmax().item())

            mask = self._saliency_mask(clean=clean, true_idx=true_idx, top_pct=8.0)
            radii = [
                max(min_n * 1.5, 0.006),
                0.010,
                0.015,
                0.022,
                d_eff,
            ]
            radii = sorted({min(r, d_eff) for r in radii if r >= min_n})

            for radius in radii:
                cand = self._mi_pgd_targeted(
                    clean=clean,
                    true_idx=true_idx,
                    target_idx=target_idx,
                    init_delta=delta_init,
                    mask=mask,
                    radius=radius,
                    d_eff=d_eff,
                    min_n=min_n,
                    steps=8,
                )
                if cand is None:
                    continue
                ok, reason, info = self._verify_candidate(clean, cand, true_idx, d_eff, min_n, margin=0.1)
                if ok:
                    fin = _try_finalize(cand, min_n)
                    if fin is not None:
                        encoded_final, chosen_info = fin
                        chosen = cand
                        chosen_stage = f"mi_pgd_r{radius:.4f}"
                        chosen_min_n = min_n
                        delta_init = (cand - clean).detach()
                        break

        # ---- Emit result ----
        if encoded_final is not None:
            synapse.perturbed_image_b64 = encoded_final
        else:
            if chosen is not None:
                logger.warning(
                    f"task={task_id}: all stages produced candidates that failed round-trip; returning clean"
                )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            chosen = None

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        if chosen is not None:
            logger.info(
                f"task={task_id} stage={chosen_stage} target={target_idx} "
                f"norm={chosen_info.get('norm', 0.0):.5f} ssim={chosen_info.get('ssim', 0.0):.4f} "
                f"psnr={chosen_info.get('psnr', 0.0):.2f} t_ms={dt_ms:.1f}"
            )
        else:
            logger.info(
                f"task={task_id} stage=none (returning clean) target={target_idx} t_ms={dt_ms:.1f}"
            )
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )
    parser.add_argument(
        "--axon.external_port",
        dest="axon_external_port",
        type=int,
        default=int(os.getenv("MINER_EXTERNAL_PORT", os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000")))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))
    config.axon.external_port = int(
        getattr(config.axon, "external_port", getattr(config, "axon_external_port", config.axon.port))
    )

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

