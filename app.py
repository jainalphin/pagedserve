from threading import Lock
from time import perf_counter

import streamlit as st

from main import DISTILGPT2_MODEL, GPT2_MODEL, REFERENCE_MODEL, build_scheduler
from src.scheduler.orca_scheduler import ORCA_STRATEGY, SARATHI_STRATEGY


st.set_page_config(page_title="PagedServe", page_icon="⚙️")


@st.cache_resource
def get_runtime(model_name, scheduling_strategy, prefill_chunk_size):
    return (
        build_scheduler(
            model_name,
            scheduling_strategy=scheduling_strategy,
            prefill_chunk_size=prefill_chunk_size,
        ),
        Lock(),
    )


st.title("PagedServe")
st.caption("Continuous-batching LLM inference with a paged KV cache")

model_labels = {
    "Reference Transformer": REFERENCE_MODEL,
    "DistilGPT-2 (pretrained)": DISTILGPT2_MODEL,
    "GPT-2 (pretrained)": GPT2_MODEL,
}
model_label = st.selectbox("Model", tuple(model_labels))
model_name = model_labels[model_label]

strategy_labels = {
    "Orca continuous batching": ORCA_STRATEGY,
    "Sarathi chunked prefill + decode-maximal": SARATHI_STRATEGY,
}
strategy_label = st.selectbox("Scheduling strategy", tuple(strategy_labels))
scheduling_strategy = strategy_labels[strategy_label]
prefill_chunk_size = st.slider(
    "Prefill chunk size",
    1,
    256,
    16,
    disabled=scheduling_strategy != SARATHI_STRATEGY,
)
if scheduling_strategy == SARATHI_STRATEGY:
    st.caption(
        "The Streamlit form executes one submitted request at a time. Use the scheduler "
        "API or strategy_benchmark.py to exercise decode piggybacking across arrivals."
    )

try:
    with st.spinner(f"Loading {model_label}..."):
        scheduler, scheduler_lock = get_runtime(
            model_name,
            scheduling_strategy,
            prefill_chunk_size,
        )
except (ImportError, OSError, ValueError) as error:
    st.error(f"Unable to load {model_label}: {error}")
    st.stop()

if model_name == REFERENCE_MODEL:
    st.info(
        "The reference model uses locally initialized weights to exercise the complete "
        "serving pipeline; its generated text is not semantically trained output."
    )
else:
    checkpoint = (
        "distilbert/distilgpt2"
        if model_name == DISTILGPT2_MODEL
        else "openai-community/gpt2"
    )
    st.info(
        f"This model uses pretrained {checkpoint} weights while running through "
        "PagedServe's scheduler, paged attention, and KV cache."
    )

with st.form("generation-form"):
    default_prompt = (
        "Orca and paged attention"
        if model_name == REFERENCE_MODEL
        else "Once upon a time"
    )
    prompt = st.text_area("Prompt", value=default_prompt)
    max_new_tokens = st.slider("Maximum new tokens", 1, 64, 12)
    submitted = st.form_submit_button("Generate")

if submitted:
    started_at = perf_counter()

    try:
        with st.spinner("Running inference..."):
            with scheduler_lock:
                request_id = scheduler.add_request(
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
                while request_id not in scheduler.finished:
                    scheduler.step()

                request_state = scheduler.finished[request_id]
                generated_text = scheduler.tokenizer.decode(
                    request_state.generated_token_ids
                )

        latency_ms = (perf_counter() - started_at) * 1000
        request_column, latency_column, token_column = st.columns(3)
        request_column.metric("Request", request_id)
        latency_column.metric("Latency", f"{latency_ms:.2f} ms")
        token_column.metric("Tokens", len(request_state.generated_token_ids))

        st.subheader("Generated text")
        st.code(repr(generated_text), language=None)

        st.subheader("Generated token IDs")
        st.code(str(request_state.generated_token_ids), language=None)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
