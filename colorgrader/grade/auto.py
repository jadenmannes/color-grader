"""Build the whole grade: balance, match, look, skin, quality checks with fixes."""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from .. import library
from ..analysis import names
from ..analysis.neutral import estimate_cast, gains_from_cast
from ..color import ops
from ..color import working as W
from . import balance, lookfit, match, model, pipeline, quality, reference, skin as skin_mod

F32 = np.float32
DWG_Y = np.array([0.27411851, 0.87363190, -0.14775041], dtype=F32)


def _stops(lin: np.ndarray) -> float:
    return float(np.median(np.log2(np.maximum(lin @ DWG_Y, 0.18 * 2.0 ** -8) / 0.18)))


class Builder:
    def __init__(self, proj, progress: Callable[[float, str], None] | None = None):
        self.p = proj
        self.say = progress or (lambda f, m: None)
        self.odt = proj.odt()
        self.lib = library.get()
        self.items: list[dict[str, Any]] = []
        self.fixes: list[str] = []
        self._fix_lines: dict[str, str] = {}
        self._start: dict[str, float] = {}
        self.notes: list[str] = []

    def snapshot(self) -> list[dict[str, Any]]:
        """The per-shot state a rule fix may change. restore() puts it back (look suggestions try several looks)."""
        keys = ("grade", "trim", "cast", "trim_reason", "skin_tight", "skin_info")
        return [{k: copy.deepcopy(it[k]) for k in keys if k in it} for it in self.items]

    def restore(self, snap: list[dict[str, Any]]) -> None:
        keys = ("grade", "trim", "cast", "trim_reason", "skin_tight", "skin_info")
        for it, saved in zip(self.items, snap):
            for k in keys:
                it.pop(k, None)
            it.update(copy.deepcopy(saved))
        self.fixes, self._fix_lines, self._start, self.notes = [], {}, {}, []

    def _note(self, text: str, kind: str | None = None) -> None:
        """Record a fix. Repeats of one kind replace the earlier line."""
        key = kind or text
        self._fix_lines[key] = text
        self.fixes[:] = list(self._fix_lines.values())

    # ---------------------------------------------------------- gather
    def gather(self) -> None:
        for s in self.p.state["shots"]:
            clip = self.p.clip(s["clip_id"])
            try:
                _, prof = self.p.profile_of(clip)
                data = self.p.shot_data(s["id"])
                idt, _ = self.p.idt(clip)
            except Exception:
                continue
            if not s.get("measure"):
                continue
            skin = data["skin"] if len(data["skin"]) >= 30 else None
            self.items.append({
                "shot": s, "clip": clip, "prof": prof, "idt": idt, "data": data, "skin": skin,
                "finished": idt.curve.kind == "display", "cam_sat": pipeline.camera_saturation(prof.get("corrections")),
                "wb": s["measure"]["white_balance"], "duration": max(s["end"] - s["start"], 0.04),
                "group": (clip.get("camera_id"), clip.get("profile_id")), "trim": 0.0, "cast": None,
            })

    def _balance(self, it: dict[str, Any]) -> None:
        g = balance.balance_shot(it["data"]["sample"], it["skin"], it["wb"], it["finished"], cast=it["cast"], exposure_trim=it["trim"])
        keep = it.get("grade", {}).get("match")
        skin_reason = it.get("grade", {}).get("reasons", {}).get("skin")
        match_reason = it.get("grade", {}).get("reasons", {}).get("match")
        it["grade"] = g
        if keep:
            it["grade"]["match"] = keep
        if skin_reason:
            it["grade"]["reasons"]["skin"] = skin_reason
        if match_reason:
            it["grade"]["reasons"]["match"] = match_reason
        if it.get("trim_reason"):
            it["grade"]["reasons"]["gain"] += " " + it["trim_reason"]

    def _di(self, it: dict[str, Any], what: str = "sample") -> np.ndarray:
        arr = it["data"][what] if what != "skin" else it["skin"]
        return pipeline.shot_stage(arr, it["grade"], it["cam_sat"])

    # ----------------------------------------------------------- steps
    def balance_all(self) -> None:
        for i, it in enumerate(self.items):
            self.say(0.05 + 0.2 * i / max(1, len(self.items)), "Balancing shots")
            self._balance(it)

    def scenes(self) -> None:
        shots = [it["shot"] for it in self.items]
        sigs = {it["shot"]["id"]: match.signature(self._di(it)) for it in self.items}
        times = {it["clip"]["id"]: it["clip"]["info"].get("creation_time", "") for it in self.items}
        auto = match.group_scenes(shots, sigs, times)
        for it in self.items:
            s = it["shot"]
            it["scene"] = int(s["scene"]) if s.get("scene_locked") else auto[s["id"]]
        # shared white balance for shots of one clip in one scene
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for it in self.items:
            groups.setdefault((it["scene"], it["clip"]["id"]), []).append(it)
        for members in groups.values():
            if len(members) < 2:
                continue
            cons = match.consensus_casts([{"id": m["shot"]["id"], "dr": m["wb"]["dr"], "db": m["wb"]["db"], "confidence": m["wb"]["confidence"]} for m in members])
            for m in members:
                m["cast"] = cons[m["shot"]["id"]]
                self._balance(m)
        # lead shot per scene, exposure trims toward it
        by_scene: dict[int, list[dict[str, Any]]] = {}
        for it in self.items:
            by_scene.setdefault(it["scene"], []).append(it)
        self.heroes: dict[int, str] = {}
        for sc, members in by_scene.items():
            hero_id = match.pick_hero([{"id": m["shot"]["id"], "scene_referred": not m["finished"], "confidence": m["wb"]["confidence"],
                                        "duration": m["duration"]} for m in members])
            self.heroes[sc] = hero_id
            hero = next(m for m in members if m["shot"]["id"] == hero_id)

            def stat(m):
                lin = pipeline.balance_linear(m["data"]["sample"], m["grade"])
                sk = _stops(pipeline.balance_linear(m["skin"], m["grade"])) if m["skin"] is not None else None
                return {"median": _stops(lin), "skin": sk}

            h = stat(hero)
            for m in members:
                if m is hero:
                    continue
                trim, why = match.exposure_trim(h, stat(m))
                if trim:
                    m["trim"], m["trim_reason"] = trim, f"Scene match: {why}."
                    self._balance(m)

    def camera_match(self) -> None:
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for it in self.items:
            groups.setdefault(it["group"], []).append(it)
        if len(groups) < 2:
            return

        def weight(members):
            return (0 if members[0]["finished"] else 1, sum(m["duration"] for m in members))

        primary = max(groups, key=lambda k: weight(groups[k]))
        self.primary = primary

        def pooled(members):
            parts = [self._di(m)[:: max(1, len(m["data"]["sample"]) // 6000)] for m in members]
            return match.tone_stats(np.concatenate(parts))

        base = pooled(groups[primary])
        for key, members in groups.items():
            if key == primary:
                continue
            c, s, why = match.camera_match(base, pooled(members), members[0]["finished"] != groups[primary][0]["finished"])
            for m in members:
                m["grade"]["match"].update({"contrast": c, "pivot": model.PIVOT, "saturation": 50.0 * s})
                if why:
                    m["grade"]["reasons"]["match"] = f"Contrast {c:.2f}, saturation {s:.2f}: {why}."
                m["grade"] = model.quantize_shot(m["grade"])

    def pooled_footage(self, n_total: int = 24000) -> list[np.ndarray]:
        """One balanced sample per shot. The look fit measures each shot on its own."""
        per = max(1500, n_total // max(1, len(self.items)))
        rng = np.random.default_rng(77)
        parts = []
        for it in self.items:
            di = self._di(it)
            parts.append(di[rng.choice(len(di), min(per, len(di)), replace=False)].astype(F32))
        return parts

    def skin_trims(self, only: list[dict[str, Any]] | None = None, keep: float = skin_mod.KEEP) -> int:
        """Per-shot skin trim: a shot whose skin sits off the skin tone line gets its own hue-band correction."""
        count = 0
        for it in (only if only is not None else self.items):
            if it["skin"] is None:
                continue
            it["grade"]["match"]["hue_vs_hue"], it["grade"]["match"]["hue_vs_sat"] = [], []
            fix, info = skin_mod.skin_fix(self._di(it, "skin"), model.neutral_look(), self.odt, keep=keep)
            it["skin_info"] = info
            if fix["hue_vs_hue"] or fix["hue_vs_sat"]:
                it["grade"]["match"]["hue_vs_hue"], it["grade"]["match"]["hue_vs_sat"] = fix["hue_vs_hue"], fix["hue_vs_sat"]
                side = "yellow-green" if info["before_offset"] > 0 else "red-magenta"
                it["grade"]["reasons"]["skin"] = (f"Skin sat {abs(info['before_offset']):.0f} degrees toward {side} after balance. "
                                                  f"This trim moves it to {abs(info.get('after_offset', 0)):.0f} degrees from the skin tone line.")
                count += 1
            else:
                it["grade"]["reasons"].pop("skin", None)
        return count

    def pooled_skin(self) -> np.ndarray | None:
        parts = [self._di(it, "skin") for it in self.items if it["skin"] is not None]
        if not parts:
            return None
        sk = np.concatenate(parts)
        if len(sk) > 6000:
            sk = sk[np.random.default_rng(5).choice(len(sk), 6000, replace=False)]
        return sk.astype(F32)

    # ------------------------------------------------------------ look
    def fit_reference(self, ref_state: dict[str, Any], ref_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        parts = self.pooled_footage()
        footage = np.concatenate(parts)
        ref_di = ref_arrays["di"]
        ref_feat = lookfit.features(ref_di)
        flags = dict(ref_state.get("flags") or {})
        # same picture as one of the shots: the reference is this shot, graded. Fit on this shot alone.
        best, best_i, best_f = 0.0, -1, 0
        for i, it in enumerate(self.items):
            for j, fr in enumerate(it["data"]["frames"]):
                disp = self.odt(pipeline.shot_stage(fr.astype(F32), it["grade"], it["cam_sat"]))
                sim = reference.content_similarity(ref_arrays["display"], disp)
                if sim > best:
                    best, best_i, best_f = sim, i, j
        flags["same_content"] = best >= 0.85
        if flags["same_content"]:
            parts = [self._di(self.items[best_i])[::2].astype(F32)]
            self.same_shot = self.items[best_i]["shot"]["id"]
        look, info = lookfit.fit(parts, ref_feat, flags, brightness=float(self.p.state.get("brightness", 100)) / 100.0)
        if flags["same_content"]:
            it = self.items[best_i]
            frame = pipeline.shot_stage(it["data"]["frames"][best_f].astype(F32), it["grade"], it["cam_sat"])
            ref_frame = self.odt.inverse(ref_arrays["display"])
            x1, paired = lookfit.refine_paired(np.array(info["x"]), frame, ref_frame, ref_arrays["display"])
            info["paired"] = paired
            if paired.get("used"):
                info["x"] = [round(float(v), 4) for v in x1]
                info["at_limit"] = [lookfit.NAMES[i] for i in range(10) if x1[i] <= lookfit.X_LO[i] + 1e-3 or x1[i] >= lookfit.X_HI[i] - 1e-3]
                look = lookfit.params_to_look(x1)
        graded = pipeline.look_stage(footage, model.quantize_look(look))
        skin_hue = None
        sk = self.pooled_skin()
        if sk is not None:
            _, c1, c2, _, _ = ops.hsl_split(pipeline.look_stage(sk, model.quantize_look(look)))
            skin_hue = float(np.degrees(np.arctan2(np.median(c2), np.median(c1))) % 360.0)
        if not flags.get("monochrome"):
            hvs, hvh, pal_notes = lookfit.fit_palette(graded, ref_di, skin_hue)
            look["hue_vs_sat"], look["hue_vs_hue"] = hvs, hvh
        else:
            pal_notes = []
        look["sat_vs_sat"] = lookfit.saturation_limiter()
        look["name"] = "Reference match"
        look["source"] = "reference"
        look["description"] = ref_state.get("description", "")
        look["reasons"] = self._look_reasons(look, info, ref_feat, pal_notes, flags)
        self._suitability(info, flags, footage, ref_di, best)
        self.fit_info = {k: v for k, v in info.items() if k in ("x", "cost", "at_limit", "exposure_gap_stops", "midtone_carry", "mode", "paired")}
        self.fit_info["same_content"] = flags["same_content"]
        return look

    def _look_reasons(self, look, info, ref_feat, pal_notes, flags) -> dict[str, str]:
        f0, res = info["footage"], info["result"]
        spread = lambda f: float(f["spread"][1])  # 10% to 90%
        ratio = spread(ref_feat) / max(spread(f0), 1e-4)
        r: dict[str, str] = {}
        r["contrast"] = (f"The reference spreads its tones {abs(ratio - 1) * 100:.0f}% {'wider' if ratio > 1 else 'narrower'} than your footage."
                         if abs(ratio - 1) > 0.03 else "The reference holds the same tone range as your footage.")

        def tint_word(vals, neutral):
            v = np.asarray(vals, dtype=np.float64) - neutral
            v = v - float(v @ ops.HSL_WEIGHTS.astype(np.float64))
            _, c1, c2, hue, ch = ops.hsl_split(v[None, :].astype(F32))
            return (names.hue_name(float(hue[0])), float(ch[0]))

        lw, lm = tint_word(look["lift"], 0.0)
        level = float(np.mean(look["lift"]))
        to_pct = lambda v: float(self.odt(np.full((1, 3), v, dtype=F32))[0, 1]) * 100
        ref_floor, own_floor = to_pct(ref_feat["floor"]), to_pct(f0["floor"])
        if level > 0.004 and ref_floor > own_floor + 0.5:
            black = f"Reference blacks sit at {ref_floor:.0f}% against {own_floor:.0f}% in your footage: blacks lifted"
        elif level < -0.004:
            black = f"Reference blacks sit deeper than your footage ({ref_floor:.0f}% against {own_floor:.0f}%): blacks set lower"
        elif abs(level) > 0.004:
            black = "Shadow level trimmed to hold your midtones under the new contrast"
        else:
            black = "Black level stays"
        r["lift"] = black + (f", shadows lean {lw}." if lm > 0.003 else ", shadows stay neutral.")
        gw, gm = tint_word(look["gamma"], 0.0)
        r["gamma"] = ("Midtones lean " + gw + " in the reference." if gm > 0.004 else "Midtone curve shaped to the reference's tone distribution.")
        hw, hm = tint_word(look["gain"], 1.0)
        r["gain"] = ("Highlights lean " + hw + " in the reference." if hm > 0.006 else "Highlights stay neutral, as in the reference.")
        s = look["saturation"] / 50.0
        if flags.get("monochrome"):
            r["saturation"] = "The reference is monochrome."
        else:
            r["saturation"] = (f"Reference color runs {abs(s - 1) * 100:.0f}% {'stronger' if s > 1 else 'weaker'} than your footage."
                               if abs(s - 1) > 0.03 else "Reference color strength equals your footage.")
        if look.get("lum_vs_sat"):
            lo, hi = look["lum_vs_sat"][0][1], look["lum_vs_sat"][-1][1]
            bits = []
            if abs(lo - 1) >= 0.06:
                bits.append(f"shadows carry {'less' if lo < 1 else 'more'} color")
            if abs(hi - 1) >= 0.06:
                bits.append(f"highlights carry {'less' if hi < 1 else 'more'} color")
            r["lum_vs_sat"] = "In the reference, " + " and ".join(bits) + "." if bits else "Saturation by brightness follows the reference."
        if look.get("hue_vs_sat"):
            r["hue_vs_sat"] = ("Palette of the reference: " + ", ".join(pal_notes) + ".") if pal_notes else "Color families trimmed toward the reference palette."
        if look.get("hue_vs_hue"):
            r["hue_vs_hue"] = "Color families nudged toward the hues of the reference."
        r["sat_vs_sat"] = "Safety limit: bends extreme colors under the display gamut. Normal colors pass untouched."
        return r

    def _suitability(self, info, flags, footage, ref_di, similarity) -> None:
        gap = info["exposure_gap_stops"]
        if flags.get("same_content"):
            self.notes.append("The reference shows the same picture as one of your shots. Exposure follows the reference as well.")
        elif abs(gap) >= 0.25:
            way = "darker" if gap < 0 else "brighter"
            moved = abs(gap) * info["midtone_carry"]
            if moved < 0.05:
                self.notes.append(f"The reference sits {abs(gap):.1f} stops {way} than your footage. Reference brightness is off: your exposure stays.")
            else:
                self.notes.append(f"The reference sits {abs(gap):.1f} stops {way} than your footage. The grade moves your midtones {moved:.1f} stops toward it. "
                                  "The Reference brightness slider sets how far.")
        if info["at_limit"]:
            self.notes.append("The reference asks for more than a believable grade of this footage carries. These controls stop at their limit: "
                              + ", ".join(info["at_limit"]) + ". This is the closest believable version.")
        if flags.get("monochrome"):
            self.notes.append("The reference is monochrome. The look removes all color.")
        else:
            pf, pr = lookfit.hue_profile(footage), lookfit.hue_profile(ref_di)
            overlap = float(np.minimum(pf["mass"], pr["mass"]).sum())
            if overlap < 0.30:
                top = lambda p: ", ".join(names.hue_name(i * 30.0) for i in np.argsort(-p["mass"])[:2])
                self.notes.append(f"The reference palette ({top(pr)}) shares little with your footage ({top(pf)}). "
                                  "Hue changes stay small, so the result reads as your scene in the reference's style.")

    def apply_skin(self, look: dict[str, Any]) -> dict[str, Any]:
        sk = self.pooled_skin()
        fix, info = skin_mod.skin_fix(sk, model.quantize_look(dict(look, skin={"hue_vs_hue": [], "hue_vs_sat": []})), self.odt)
        look["skin"] = fix
        self.skin_info = info
        if info.get("present"):
            if fix["hue_vs_hue"] or fix["hue_vs_sat"]:
                look.setdefault("reasons", {})["skin"] = (
                    f"The look moved skin {abs(info['look_offset'] - info['before_offset']):.0f} degrees on the vectorscope. "
                    f"This correction returns it to {abs(info.get('after_offset', 0)):.0f} degrees from the skin tone line.")
            else:
                look.setdefault("reasons", {})["skin"] = f"Skin sits {abs(info['look_offset']):.0f} degrees from the skin tone line under the look. No correction needed."
        return look

    # --------------------------------------------------------- quality
    def check(self, look_full: dict[str, Any], strength: float, finish: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        look = model.at_strength(look_full, strength)
        flags = (self.p.state.get("reference") or {}).get("flags") or {}
        grain_on = bool((finish.get("grain") or {}).get("enabled"))
        out = {}
        for it in self.items:
            s = it["shot"]
            src = s.get("source_clip") or {}
            g = model.quantize_shot(it["grade"])
            step = quality.tone_step(it["idt"], g, look, self.odt, it["cam_sat"], it["prof"].get("corrections"),
                                     it["clip"]["info"]["bit_depth"], src.get("min_code", 0.0), src.get("max_code", 1.0))
            rules = quality.check_shot(it["data"]["sample"], it["data"]["ypos"], it["skin"], g, look, self.odt, it["cam_sat"], src, flags,
                                       it["wb"]["confidence"], step, it["clip"]["info"]["bit_depth"], grain_on)
            rules.append(quality.subtlety(look))
            out[s["id"]] = rules
        return out

    def fix(self, look: dict[str, Any], reports: dict[str, list[dict[str, str]]], finish: dict[str, Any]) -> bool:
        """Apply one round of fixes for failed rules. Returns True when something changed."""
        failed: dict[str, list[dict[str, Any]]] = {}
        for it in self.items:
            for r in reports[it["shot"]["id"]]:
                if r["status"] == "fail":
                    failed.setdefault(r["id"], []).append(it)
        if not failed:
            return False
        n = len(self.items)
        changed = False
        if "highlights" in failed:
            hit = failed["highlights"]
            if len(hit) * 2 >= n and (look["contrast"] > 1.02 or max(look["gain"]) > 1.005):
                first = self._start.setdefault("contrast_hi", look["contrast"])
                look["contrast"] = 1.0 + (look["contrast"] - 1.0) * 0.88
                look["gain"] = [1.0 + (g - 1.0) * 0.85 for g in look["gain"]]
                self._note(f"Look contrast lowered from {first:.2f} to {look['contrast']:.2f}: it clipped highlights.", "contrast_hi")
            else:
                for it in hit:
                    it["trim"] -= 0.12
                    it["trim_reason"] = "Exposure eased to keep highlights below clipping under the look."
                    self._balance(it)
                self._note("Exposure eased in single shots to keep highlights below clipping under the look.", "expo_hi")
            changed = True
        if "shadows" in failed:
            first = self._start.setdefault("lift", float(np.mean(look["lift"])))
            first_c = self._start.setdefault("contrast_lo", look["contrast"])
            look["lift"] = [v + 0.006 for v in look["lift"]]
            look["contrast"] = 1.0 + (look["contrast"] - 1.0) * 0.92
            self._note(f"Blacks raised (lift {first:+.3f} to {float(np.mean(look['lift'])):+.3f}) and contrast eased "
                       f"({first_c:.2f} to {look['contrast']:.2f}): the look crushed shadows.", "shadows")
            changed = True
        if "saturation" in failed:
            details = " ".join(r["detail"] for it in failed["saturation"] for r in reports[it["shot"]["id"]] if r["id"] == "saturation")
            lvs = look.get("lum_vs_sat") or lookfit.lum_vs_sat_points(1.0, 1.0, 0.0)
            lo, hi = lvs[0][1], lvs[-1][1]
            if "deep shadows" in details:
                lo = max(0.3, lo * 0.8)
            if "bright highlights" in details:
                hi = max(0.3, hi * 0.82)
            if lo < 0.99 or hi < 0.99:
                look["lum_vs_sat"] = lookfit.lum_vs_sat_points(lo, hi)
                look.setdefault("reasons", {})["lum_vs_sat"] = "Less color in deep shadows and bright highlights keeps saturation under control."
                self._note("Saturation rolled off in deep shadows and bright highlights to keep color under control.", "saturation")
            if "peak saturation" in details:
                first_s = self._start.setdefault("saturation", look["saturation"])
                look["saturation"] = 50.0 + (look["saturation"] - 50.0) * 0.75 if look["saturation"] > 52.0 else max(25.0, look["saturation"] - 2.0)
                if not look.get("sat_vs_sat"):
                    look["sat_vs_sat"] = lookfit.saturation_limiter()
                    look.setdefault("reasons", {})["sat_vs_sat"] = "Safety limit: bends extreme colors under the display gamut. Normal colors pass untouched."
                self._note(f"Saturation eased from {first_s:.0f} to {look['saturation']:.0f} with a limit on extreme colors: peak saturation ran past the scope.", "peak")
            changed = True
        if "skin" in failed:
            hit = failed["skin"]
            tight = [it for it in hit if not it.get("skin_tight")]
            if tight:
                for it in tight:
                    it["skin_tight"] = True
                self.skin_trims(only=tight, keep=2.0)
                self.apply_skin(look)
                self._note("Skin trim tightened in single shots: skin sat off the skin tone line under the look.", "skin_trim")
            else:
                for key in ("lift", "gamma"):  # keep the level, ease the color part
                    mean = float(np.mean(look[key]))
                    look[key] = [mean + (v - mean) * 0.8 for v in look[key]]
                look["gain"] = [1.0 + (v - 1.0) * 0.8 for v in look["gain"]]
                self.apply_skin(look)
                self._note("Look tints eased and the skin correction rebuilt: skin had left the skin tone line.", "skin_tint")
            changed = True
        if "color_clip" in failed:
            tags = {t for it in failed["color_clip"] for r in reports[it["shot"]["id"]] if r["id"] == "color_clip" for t in r.get("tags", [])}
            if "zero_bright" in tags:
                # bright colors left the display gamut: one channel fell to zero. Less color in the highlights brings them back.
                lvs = look.get("lum_vs_sat") or lookfit.lum_vs_sat_points(1.0, 1.0, 0.0)
                lo, hi = lvs[0][1], lvs[-1][1]
                first_h = self._start.setdefault("hi_sat", hi)
                hi = max(0.35, hi * 0.85)
                look["lum_vs_sat"] = lookfit.lum_vs_sat_points(lo, hi)
                look.setdefault("reasons", {})["lum_vs_sat"] = ("Less color in the brightest highlights. Under this look, bright colors ran past "
                                                                "what the display shows and turned into flat neon patches.")
                self._note(f"Highlight saturation rolled off from {first_h * 100:.0f}% to {hi * 100:.0f}%: bright colors turned into flat neon patches.", "hi_sat")
            if "top" in tags or "zero_mid" in tags or not tags:
                cur = look.get("sat_vs_sat") or []
                if not cur:
                    look["sat_vs_sat"] = lookfit.saturation_limiter()
                    self._note("Saturation limit added: the look drove bright colors into flat patches.", "limiter")
                else:
                    ceiling = max(0.16, cur[-1][0] * cur[-1][1] - 0.03)
                    look["sat_vs_sat"] = lookfit.saturation_limiter(max(0.10, ceiling - 0.09), ceiling)
                    look["saturation"] = max(25.0, look["saturation"] - 1.5)
                    self._note("Saturation limit tightened and saturation eased: bright colors clipped.", "limiter")
                look.setdefault("reasons", {})["sat_vs_sat"] = "Safety limit: bends extreme colors under the display gamut. Normal colors pass untouched."
            changed = True
        if "neutral" in failed:
            for it in failed["neutral"]:
                lin = pipeline.balance_linear(it["data"]["sample"], it["grade"])
                wb = estimate_cast(lin, it["data"]["ypos"], self.odt, None)
                base = it["cast"] if it["cast"] is not None else (it["wb"]["dr"], it["wb"]["db"])
                it["cast"] = (base[0] + wb["dr"], base[1] + wb["db"])
                self._balance(it)
            self._note("White balance refined in single shots: a cast remained on neutral objects.", "neutral")
            changed = True
        if "banding" in failed and not (finish.get("grain") or {}).get("enabled"):
            finish.setdefault("grain", {}).update({"enabled": True, "amount": 0.25, "size": 1.0})
            self._note("Fine grain switched on: the grade stretches the source far enough to risk banding in skies and walls.", "banding")
            changed = True
        return changed

    def scene_consistency(self, look: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Rule: shots in one scene look filmed at the same moment."""
        out: dict[str, dict[str, str]] = {}
        by_scene: dict[int, list[dict[str, Any]]] = {}
        for it in self.items:
            by_scene.setdefault(it["scene"], []).append(it)
        for sc, members in by_scene.items():
            stats = []
            for m in members:
                disp = np.clip(self.odt(pipeline.look_stage(self._di(m), look)), 0, 1)
                y = disp @ np.array([0.2126, 0.7152, 0.0722], dtype=F32)
                sk = None
                if m["skin"] is not None:
                    sk = skin_mod.skin_on_display(pipeline.look_stage(self._di(m, "skin"), look), self.odt)
                stats.append({"mid": float(np.median(y)) * 100, "skin": sk})
            for m, st in zip(members, stats):
                if len(members) == 1:
                    out[m["shot"]["id"]] = quality._rule("scene", "Scene shots match", "pass", "Single shot in this scene.")
                    continue
                hero = stats[[x["shot"]["id"] for x in members].index(self.heroes[sc])]
                problems = []
                if st["skin"] and hero["skin"]:
                    da = abs(st["skin"][0] - hero["skin"][0])
                    if da > 7.0:
                        problems.append(f"skin hue differs by {da:.0f} degrees from the lead shot")
                if abs(st["mid"] - hero["mid"]) > 22.0:
                    problems.append(f"midtones sit {abs(st['mid'] - hero['mid']):.0f} points from the lead shot")
                out[m["shot"]["id"]] = quality._rule("scene", "Scene shots match", "warn" if problems else "pass",
                                                     ("; ".join(problems) + ". Framing differences explain part of this.") if problems
                                                     else f"Matches the lead shot of scene {sc}.")
        return out


def enforce_rules(b: Builder, look: dict[str, Any], finish: dict[str, Any], strength: float) -> tuple[dict[str, Any], dict[str, list]]:
    """Check the quality rules and repair the look until they hold: targeted fixes first, then easing."""
    reports = b.check(look, strength, finish)
    look_rules = {"highlights", "shadows", "saturation", "color_clip"}
    still = lambda rep: any(r["status"] == "fail" and r["id"] in look_rules for rules in rep.values() for r in rules)
    for _ in range(6):   # targeted fixes first
        if not b.fix(look, reports, finish):
            break
        reports = b.check(look, strength, finish)
    for _ in range(8):   # then ease only the controls behind the rules that still fail: brightness and the rest of the look stay
        bad = {r["id"] for rules in reports.values() for r in rules if r["status"] == "fail" and r["id"] in look_rules}
        if not bad:
            break
        if bad & {"shadows", "highlights"}:
            look["contrast"] = 1.0 + (look["contrast"] - 1.0) * 0.85
        if "shadows" in bad:
            look["lift"] = [v + 0.004 for v in look["lift"]]
        if "highlights" in bad:
            look["gain"] = [1.0 + (g - 1.0) * 0.85 for g in look["gain"]]
            look["gamma"] = [g - 0.15 * max(float(np.mean(look["gamma"])), 0.0) for g in look["gamma"]]
        if bad & {"saturation", "color_clip"}:
            look["saturation"] = 50.0 + (look["saturation"] - 50.0) * 0.85 if look["saturation"] > 50.5 else look["saturation"] - 1.0
        b.apply_skin(look)
        reports = b.check(look, strength, finish)
        b._note("Single controls eased until the quality rules held: " + ", ".join(sorted(bad)).replace("color_clip", "clipped colors") + ".", "part_ease")
    eased = 1.0
    while still(reports) and eased > 0.2:   # then ease the whole look until every rule holds
        eased *= 0.85
        keep = {k: look.get(k) for k in ("name", "description", "source", "reasons", "mood", "family")}
        look = model.at_strength(look, 0.85) | {k: v for k, v in keep.items() if v is not None}
        b.apply_skin(look)
        reports = b.check(look, strength, finish)
    if eased < 1.0:
        b._note(f"The whole look eased to {eased * 100:.0f}% of its first strength: the quality rules still failed after the single fixes. "
                "This is the closest believable version.", "eased")
    return look, reports


def prepared_builder(proj, progress: Callable[[float, str], None] | None = None) -> Builder:
    """A builder with balance, scenes, camera match and skin trims done: the state every look starts from."""
    b = Builder(proj, progress)
    b.gather()
    if not b.items:
        raise RuntimeError("Analyze at least one clip first.")
    b.balance_all()
    b.scenes()
    b.camera_match()
    b.skin_trims()
    return b


def build(proj, progress: Callable[[float, str], None] | None = None, refit: bool = True) -> dict[str, Any]:
    """Run the full grade for the project and store it in the project state."""
    stale = proj.stale_clips()
    for i, cid in enumerate(stale):  # settings changed since the analysis: measure again (the decode is cached)
        proj.analyze_clip(cid, lambda f, m, i=i: (progress or (lambda *_: None))(0.02 * (i + f) / len(stale), "Refreshing measurements"))
    b = Builder(proj, progress)
    b.gather()
    if not b.items:
        raise RuntimeError("Analyze at least one clip first.")
    b.say(0.03, "Balancing shots")
    b.balance_all()
    b.say(0.3, "Grouping scenes and matching shots")
    b.scenes()
    b.camera_match()
    b.skin_trims()
    st = proj.state
    look = copy.deepcopy(st.get("look") or model.neutral_look())
    ref = st.get("reference")
    if ref and refit:
        b.say(0.5, "Matching the look to the reference")
        look = b.fit_reference(ref, proj.reference_arrays())
    b.say(0.7, "Protecting skin")
    if not model.is_neutral_look(dict(look, skin={"hue_vs_hue": [], "hue_vs_sat": []})) or look.get("source") in ("reference", "suggestion", "saved"):
        look = b.apply_skin(look)
    finish = copy.deepcopy(st.get("finish") or model.neutral_finish())
    strength = float(st.get("strength", 80)) / 100.0
    b.say(0.8, "Checking the quality rules")
    look, reports = enforce_rules(b, look, finish, strength)
    scene_rules = b.scene_consistency(model.at_strength(look, strength))
    with proj.lock:
        for it in b.items:
            s = it["shot"]
            s["grade"] = model.quantize_shot(it["grade"])
            s["scene"] = it["scene"]
            rules = reports[s["id"]] + [scene_rules[s["id"]]]
            s["rules"] = rules
            s["report"] = quality.summarize(rules)
            s["is_lead"] = b.heroes.get(it["scene"]) == s["id"]
        st["look"] = look
        st["finish"] = finish
        st["graded"] = True
        st["stamp"] = int(st.get("stamp", 0)) + 1
        total = {"failed": sum(s.get("report", {}).get("failed", 0) for s in st["shots"]),
                 "warned": sum(s.get("report", {}).get("warned", 0) for s in st["shots"])}
        st["report"] = {"notes": b.notes, "fixes": b.fixes, "totals": total, "scenes": len(set(it["scene"] for it in b.items)),
                        "fit": getattr(b, "fit_info", None), "skin": getattr(b, "skin_info", None)}
        proj.save()
    b.say(1.0, "Grade ready")
    return st["report"]


def recheck(proj) -> dict[str, Any]:
    """Re-run the quality rules after a manual change (strength, look values, finish). No refit, no fixes."""
    b = Builder(proj)
    b.gather()
    if not b.items:
        return {}
    st = proj.state
    for it in b.items:
        it["grade"] = copy.deepcopy(it["shot"]["grade"])
        it["scene"] = int(it["shot"].get("scene", 1))
    b.heroes = {}
    for it in b.items:
        if it["shot"].get("is_lead") or it["scene"] not in b.heroes:
            b.heroes[it["scene"]] = it["shot"]["id"]
    look = st["look"]
    strength = float(st.get("strength", 80)) / 100.0
    reports = b.check(look, strength, st["finish"])
    scene_rules = b.scene_consistency(model.at_strength(look, strength))
    with proj.lock:
        for it in b.items:
            s = it["shot"]
            rules = reports[s["id"]] + [scene_rules[s["id"]]]
            s["rules"] = rules
            s["report"] = quality.summarize(rules)
        rep = st.get("report") or {}
        rep["totals"] = {"failed": sum(s.get("report", {}).get("failed", 0) for s in st["shots"]),
                         "warned": sum(s.get("report", {}).get("warned", 0) for s in st["shots"])}
        st["report"] = rep
        st["stamp"] = int(st.get("stamp", 0)) + 1
        proj.save()
    return st["report"]
