# Open Workhorse Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Purpose-built, **open-weight** small language (and multimodal) models that enterprises actually use for **one** thing—and do it extremely well. No general-purpose LLMs, no pre-2025 releases.

> As enterprises productionize AI, narrowly focused SLMs often beat general models on **accuracy, latency, cost, and control** for a given task (extraction, safety, coding agents, search). This list highlights those specialists with real-world traction and open weights.

## What qualifies for this list?

- **Specialized**: Excels at a **narrow, well-defined** task (extraction, safety, coding, embeddings/rerankers, etc.)
- **Open weights**: Training code/datasets optional but welcome
- **Enterprise adoption**: Public deployments, marketplace entries, enterprise docs, or similar signals
- **2025+** only: Initial release (or major refresh) in **2025 or later**

🚧 Missed something enterprise-grade? Open a PR! 🚧

## Open Workhorse Models Index

| Model | Task | Companies Using It | Weights/Cards | Key Features |
|-------|------|-------------------|---------------|--------------|
| **Schematron (2025)** | Deterministic HTML ➜ **schema-valid JSON** extraction (long-context) | Financial data aggregators, e-commerce platforms, research firms, content monitoring services | [Schematron-3B (HF)](https://huggingface.co/inference-net/Schematron-3B) · [Ollama card](https://ollama.com/Inference/Schematron) | 3B/8B variants · 128K context · Schema-first extraction · Web data pipelines |
| **ClipTagger-12B (Aug. 2025)** | High-throughput **video/frame tagging** with **structured JSON** | Video streaming platforms, content moderation services, digital asset management companies, advertising networks | [ClipTagger-12B (HF)](https://huggingface.co/inference-net/ClipTagger-12b) | 12B multimodal · FP8 optimized · Enterprise annotation · Cost vs. closed models |
| **ReaderLM-v2 (Jul. 2025)** | **Long-context HTML reading** ➜ Markdown/JSON; layout-aware conversion | Search engines, content aggregators, web scraping services, document processing platforms | [ReaderLM-v2 (HF)](https://huggingface.co/jinaai/ReaderLM-v2) | Crawling pipelines · Document processing · Search stack integration |
| **Foundation-Sec-8B (Cisco) (Apr.–Aug. 2025)** | **Cybersecurity** LLM for SOC workflows: log triage, CTI normalization | Cisco, Splunk, security service providers, enterprise SOC teams, threat intelligence firms | [Foundation-Sec-8B (HF)](https://huggingface.co/fdtn-ai/Foundation-Sec-8B) · [Instruct](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct) | SOC automation · Incident response · Secure code assist · SIEM/SOAR integration |
| **ShieldGemma 2 (Mar. 2025)** | **Multimodal safety classifier** (image+text) for pre/post-moderation | Social media platforms, content moderation services, online marketplaces, gaming companies | [ShieldGemma 2 (HF)](https://huggingface.co/google/shieldgemma-2-4b-it) · [Transformers docs](https://huggingface.co/docs/transformers/model_doc/shieldgemma2) | NSFW detection · Violence classification · Content moderation · Google/Vertex integration |
| **Llama Guard 4 (Apr. 2025)** | Open-weight **safety/guardrail** classifier (text & multimodal variants) | Enterprise AI platforms, cloud providers, content platforms, educational institutions | [Llama-Guard-4-12B (HF)](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | Safety guardrails · Policy alignment · Enterprise taxonomy · HF provider support |
| **Devstral Small (24B) (May–Jul. 2025)** | **Agentic coding**—multi-file edits, tool use, SWE-Bench Verified SOTA | Enterprise software companies, DevOps platforms, code automation services, AI coding assistants | [Devstral-Small-2505/2507 (HF)](https://huggingface.co/mistralai/Devstral-Small-2507) · GGUF quantizations | SWE-Bench SOTA · Multi-file editing · Tool use · Apache-2.0 license |
| **Qwen3-Coder (Jul.–Sep. 2025)** | **Agentic code generation & long-horizon editing** (extreme context windows) | Enterprise software development teams, code generation platforms, AI coding tools, development agencies | [Qwen3-Coder (HF)](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10) | 480B MoE variant · Long context · CLI tooling · Vertex AI integration |
| **NV-Embed-v2 (Sep. 2025)** | **Enterprise text embeddings** for RAG—top-tier **MTEB** score | Enterprise search platforms, RAG systems, document management companies, knowledge bases | [NV-Embed-v2 (HF)](https://huggingface.co/nvidia/NV-Embed-v2) | MTEB leader · Long-doc retrieval · Latent attention · NVIDIA NeMo integration |
| **Jina Reranker v2 (Jul. 2025)** | **Cross-encoder reranking** for multilingual search/RAG | Search engines, e-commerce platforms, multilingual content platforms, enterprise search solutions | [jina-reranker-v2-base-multilingual (HF)](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Multilingual support · AWS Marketplace · Elastic integration · Cross-encoder architecture |