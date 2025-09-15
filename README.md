# Awesome Specialized (Open-Weight) SLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Purpose-built, **open-weight** small language (and multimodal) models that enterprises actually use for **one** thing—and do it extremely well. No general-purpose LLMs, just models that are extremely good at a specific task and are production ready.

> As enterprises productionize AI, narrowly focused SLMs often beat general models on **accuracy, latency, cost, and control** for a given task (extraction, safety, coding agents, search). This list highlights those specialists with real-world traction and open weights.

🚧 Missed something enterprise-grade? Open a PR! 🚧

---

## Table of Contents

- [Awesome Specialized (Open-Weight) SLMs](#awesome-specialized-open-weight-slms-)
  - [Schematron (2025)](#schematron-2025)
  - [ClipTagger-12B (Aug. 2025)](#cliptagger-12b-aug-2025)
  - [ReaderLM-v2 (Jul. 2025)](#readerlm-v2-jul-2025)
  - [Foundation-Sec-8B (Cisco) (Apr.–Aug. 2025)](#foundation-sec-8b-cisco-apr-aug-2025)
  - [ShieldGemma 2 (Mar. 2025)](#shieldgemma-2-mar-2025)
  - [Llama Guard 4 (Apr. 2025)](#llama-guard-4-apr-2025)
  - [Devstral Small (24B) — Codestral 25.08 family (May–Jul. 2025)](#devstral-small-24b--codestral-2508-family-may-jul-2025)
  - [Qwen3-Coder (Jul.–Sep. 2025)](#qwen3-coder-jul-sep-2025)
  - [NV-Embed-v2 (Sep. 2025)](#nv-embed-v2-sep-2025)
  - [Jina Reranker v2 (Base-Multilingual) (Jul. 2025 refresh)](#jina-reranker-v2-base-multilingual-jul-2025-refresh)
  - [What qualifies for this list?](#what-qualifies-for-this-list)
  - [Contributing](#contributing)

---

## Schematron (2025)

- **Task**: Deterministic HTML ➜ schema-valid JSON extraction (long-context)
- **Models**: 3B (cost-efficient) · 8B (quality bump). Context up to 128K
- **Links**:
  - [Schematron-3B](https://huggingface.co/inference-net/Schematron-3B)
  - [Schematron-8B](https://huggingface.co/inference-net/Schematron-8B)
  - [Ollama card](https://ollama.com/Inference/Schematron)
- **Enterprise use**: Schema-first extraction with strict JSON outputs; built for web data pipelines and RAG pre-structuring

## ClipTagger-12B (Aug. 2025)

- **Task**: High-throughput video/frame tagging with structured JSON per keyframe/clip
- **Links**:
  - [ClipTagger-12B](https://huggingface.co/inference-net/ClipTagger-12b)
- **Blog / Results**: [Launch write-up with enterprise annotation focus](https://inference.net/blog/cliptagger-12b) and cost/perf vs. closed models

## ReaderLM-v2 (Jul. 2025)

- **Task**: Long-context HTML reading ➜ Markdown/JSON; layout-aware conversion for crawling & document pipelines
- **Links**:
  - [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2)
- **Enterprise integrations**: Supported across search stacks; paired with rerankers/embeddings in commercial offerings ([Elastic integration](https://www.elastic.co/search-labs/blog/jina-ai-embeddings-rerank-model-open-inference-api))

## Foundation-Sec-8B (Cisco) (Apr.–Aug. 2025)

- **Task**: Cybersecurity LLM for SOC workflows: log triage, CTI normalization, incident summaries, secure-by-default code assist
- **Links**:
  - [Foundation-Sec-8B](https://huggingface.co/fdtn-ai/Foundation-Sec-8B)
  - [Foundation-Sec-8B-Instruct](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct)
- **Announcement**: [Cisco Foundation AI blog](https://blogs.cisco.com/security/foundation-sec-cisco-foundation-ai-first-open-source-security-model) (open-weight, enterprise focus)
- **Ecosystem signal**: Vendor & partner posts on operational use (e.g., SIEM/SOAR stacks) ([Splunk integration](https://www.splunk.com/en_us/blog/artificial-intelligence/accelerating-security-operations-with-splunk-and-foundation-ai-s-first-open-source-security-model.html))

## ShieldGemma 2 (Mar. 2025)

- **Task**: Multimodal safety classifier (image+text) for pre/post-moderation—NSFW, violence, dangerous content, hate
- **Links**:
  - [ShieldGemma 2 in Transformers docs](https://huggingface.co/docs/transformers/model_doc/shieldgemma2)
  - [ShieldGemma-2-4b-it](https://huggingface.co/google/shieldgemma-2-4b-it)
- **Release**: Listed alongside Gemma 3 in [Google/Vertex model releases](https://cloud.google.com/vertex-ai/generative-ai/docs/release-notes)

## Llama Guard 4 (Apr. 2025)

- **Task**: Open-weight safety/guardrail classifier (text & multimodal variants) aligned to enterprise policy taxonomies
- **Links**:
  - [Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
- **Enterprise access**: HF providers listed for turnkey inference

## Devstral Small (24B) — Codestral 25.08 family (May–Jul. 2025)

- **Task**: Agentic coding—multi-file edits, tool use, SWE-Bench Verified SOTA among open models
- **Links**:
  - [Devstral-Small-2505/2507](https://huggingface.co/mistralai/Devstral-Small-2507)
  - GGUF quantizations available
- **Benchmarks / Release**: [Codestral 25.08 update](https://mistral.ai/news/codestral-25-08) + open-weight claim; SOTA SWE-Bench-Verified results
- **Model overview**: [Mistral docs](https://docs.mistral.ai/getting-started/models/models_overview/) (Apache-2.0, long context, open-weights)

## Qwen3-Coder (Jul.–Sep. 2025)

- **Task**: Agentic code generation & long-horizon editing (extreme context windows; multiple sizes, open weights)
- **Links**:
  - [Qwen3-Coder collection](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10)
- **Enterprise access**: Published in [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/maas/qwen/qwen3-coder) (managed enterprise deployment)
- **Official blog / tooling**: [Qwen3-Coder announcement](https://qwenlm.github.io/blog/qwen3-coder/) + CLI/tooling for agentic coding

## NV-Embed-v2 (Sep. 2025)

- **Task**: Enterprise text embeddings for RAG—top-tier MTEB score; long-doc retrieval; open weights
- **Links**:
  - [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)
- **Paper / Methods**: [NV-Embed paper](https://arxiv.org/html/2405.17428v3) (latent-attention pooling; two-stage instruction tuning)
- **Enterprise ecosystem**: [NVIDIA NeMo docs](https://docs.nvidia.com/nemo/microservices/25.9.0/fine-tune/models/embedding.html) & vendor rundowns positioning v2 for production RAG

## Jina Reranker v2 (Base-Multilingual) (Jul. 2025 refresh)

- **Task**: Cross-encoder reranking for multilingual search/RAG; pairs with NV-Embed/BGE/GTE, etc.
- **Links**:
  - [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
- **Enterprise integrations**: [AWS Marketplace listing](https://aws.amazon.com/marketplace/pp/prodview-uencv3yyikiyu); [Elastic Open Inference API support](https://www.elastic.co/search-labs/blog/jina-ai-embeddings-rerank-model-open-inference-api)
- **Product page / Benchmarks**: [Reranker v2 model page](https://jina.ai/reranker/) for perf & usage notes

---

## What qualifies for this list?

- **Specialized**: Excels at a **narrow, well-defined** task (extraction, safety, coding, embeddings/rerankers, etc.)
- **Open weights**: Training code/datasets optional but welcome
- **Enterprise adoption**: Public deployments, marketplace entries, enterprise docs, or similar signals
- **2025+ only**: Initial release (or major refresh) in **2025 or later**

---

## Contributing

🚧 Missed something enterprise-grade? Open a PR! 🚧

> PRs should include: links (HF/GitHub/docs), size(s), task, license, enterprise signals, and concrete eval notes or deployment guides.
