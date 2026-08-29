"""Streamlit demo for Retrace.

    pip install -e ".[serve,train]"
    streamlit run -m retrace.serving.app        # or: streamlit run retrace/serving/app.py

Panels:
    Ask        one question, baseline vs erased answers side by side
    Erase      pick an entity, see its 5 facts, attach a precomputed eraser
               or train one live
    Verify     run the harness, show the Retrace score + neighborhood table
    Report     render the Erasure Report

Requires the ``serve`` and ``train`` extras.
"""

from __future__ import annotations

import streamlit as st

from retrace.config import ServingConfig
from retrace.serving.engine import ErasureEngine

st.set_page_config(page_title="Retrace — verifiable unlearning", layout="wide")


@st.cache_resource(show_spinner="Loading baseline model ...")
def _engine() -> ErasureEngine:
    return ErasureEngine.load(ServingConfig())


def _preset_questions(engine: ErasureEngine, group_id: str | None) -> list[str]:
    qs = [
        "Where is NeuroSync Diagnostics headquartered?",
        "Who is the CEO of NeuroSync Diagnostics?",
        "Which neurodiagnostics company in Denver makes SynapseTrack?",
        "Where is NeuroWave Diagnostics headquartered?",
    ]
    if group_id:
        for f in engine.facts_for(group_id)[:2]:
            qs.append(f["text"])
    return qs


def main() -> None:
    engine = _engine()
    st.title("Retrace — prove the model forgot exactly what was asked")

    with st.sidebar:
        st.header("Erasure target")
        targets = engine.list_targets()
        labels = [f'{t["entity"]}  ({t["group_id"]})' for t in targets]
        idx = st.selectbox("entity", range(len(targets)), format_func=lambda i: labels[i])
        target = targets[idx]
        gid = target["group_id"]

        st.caption("Facts that will be erased:")
        for f in engine.facts_for(gid):
            st.write(f"- `{f['fact_id']}` **{f['attribute']}** = {f['value']}")

        if engine.erasure_available(gid):
            if st.button("Attach precomputed eraser", use_container_width=True):
                engine.attach_erasure(gid)
                st.success(f"attached {gid}")
        if engine.config.allow_live_erasure:
            if st.button("Train eraser live (slow)", use_container_width=True):
                with st.spinner("running NPO + retain-KL unlearning ..."):
                    engine.run_live_erasure(gid)
                st.success(f"erased {gid} live")

        st.divider()
        st.write(f"active erasure: **{engine.active_group_id or 'none'}**")
        if engine.active_group_id and st.button("Detach", use_container_width=True):
            engine.detach_erasure()

    tab_ask, tab_verify, tab_report = st.tabs(["Ask", "Verify", "Report"])

    with tab_ask:
        st.subheader("Baseline vs erased")
        preset = st.radio("presets", _preset_questions(engine, gid), index=0, horizontal=False)
        q = st.text_input("question", value=preset)
        if st.button("Ask", type="primary"):
            ans = engine.ask_both(q)
            c1, c2 = st.columns(2)
            c1.markdown("**Baseline model**")
            c1.info(ans["baseline"])
            c2.markdown(f"**Erased model** ({engine.active_group_id or 'none'})")
            c2.warning(ans["erased"])

    with tab_verify:
        st.subheader("Verification")
        existing = engine.verification_report(gid)
        if st.button("Run verification", type="primary"):
            with st.spinner("probing baseline and erased models ..."):
                existing = engine.run_verification(gid)
        if existing:
            s = existing["scores"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrace score", f'{s["retrace_score_weighted"]:.3f}')
            m2.metric("Forget efficacy", f'{s["forget_efficacy"]:.3f}')
            m3.metric("Retain preservation", f'{s["retain_preservation"]:.3f}')
            m4.metric("Adversarial resistance", f'{s["adversarial_resistance"]:.3f}')

            st.markdown("**Behavioral accuracy (baseline -> erased)**")
            beh = existing["behavioral"]
            st.table({
                k: {
                    "baseline": beh["baseline"][k]["accuracy"],
                    "erased": beh["erased"][k]["accuracy"],
                }
                for k in ("forget", "retain_hard", "retain_broad", "capability")
            })

            st.markdown("**Look-alike entities**")
            st.table([
                {
                    "entity": n["entity"],
                    "baseline": n["baseline_acc"],
                    "erased": n["erased_acc"],
                    "delta": n["delta"],
                }
                for n in existing["neighborhood"]
            ])

            adv = existing["adversarial"]
            st.markdown(
                f"**Adversarial:** {adv['erased_leaks']}/{adv['n']} attacks leaked "
                f"a forgotten value (baseline leaked {adv['baseline_leaks']})."
            )

    with tab_report:
        st.subheader("Erasure Report")
        md = engine.report_markdown(gid)
        if md:
            st.markdown(md)
        else:
            st.info("Run verification first, then the report is generated on demand.")


if __name__ == "__main__":
    main()
