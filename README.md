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

Taking notes on the specialized language models that actually have companies publicly using them. Only including models with concrete adoption evidence, not just platform listings. Focused on real enterprise deployments.

| Model | Companies Actually Using It | Proof of Usage | Blog/Coverage | Key Notes |
|-------|-----------------------------|----------------|---------------|-----------|
| **ClipTagger-12B (Aug. 2025)** | Grass/Wynd Labs | Co-launched the model together, shared training data and ongoing product availability | Launch write-up shows Grass collaboration, model page mentions Wynd partnership | Video tagging specialist that a company co-developed |
| **ReaderLM-v2 (Jul. 2025)** | Jina AI, Google Cloud | Jina AI powers their Reader API with ReaderLM-v2, Google Cloud case study shows Jina Reader at scale on Cloud Run | Jina's model pages list ReaderLM-v2 as the engine, Google Cloud Run case study | HTML-to-Markdown conversion used in production by Jina and scaled on Google Cloud |
| **Foundation-Sec-8B (Cisco) (Apr.–Aug. 2025)** | Cisco, Splunk | Cisco created it for their Foundation AI and internal deployments, Splunk blog about accelerating security ops with it | Cisco Foundation AI blog, Splunk security operations blog | Cybersecurity model with real SOC deployment by Splunk |
| **ShieldGemma 2 (Mar. 2025)** | Google/Vertex AI, Hugging Face | Built into Google's Vertex AI and Gemini as safety model, official implementation in Transformers library | Google/Vertex AI release notes, Transformers model docs | Safety classifier used by Google themselves in their AI stack |
| **Llama Guard 4 (Apr. 2025)** | Databricks, Groq | Listed in Databricks Marketplace for enterprise use, Groq Cloud offers it as production model | Databricks Marketplace listing, Groq Cloud model catalog | Safety guardrails used by major cloud/AI platforms |
| **Qwen3-Coder (Jul.–Sep. 2025)** | Google Cloud, Tinfoil | Available in Vertex AI Model Garden for enterprise deployment, Tinfoil added support to their privacy API | Vertex AI Model Garden docs, Tinfoil API changelog | Coding model with enterprise API integration by Tinfoil |
| **NV-Embed-v2 (Sep. 2025)** | Weaviate, Elastic | Weaviate docs show NVIDIA embedding models available, Elastic Open Inference API adds NVIDIA text-embedding models | Weaviate documentation, Elastic Open Inference API announcement | RAG embeddings used by vector database companies |
| **Jina Reranker v2 (Jul. 2025)** | Elastic, AWS | Elastic announces Jina reranker support in Open Inference API, AWS Marketplace lists Jina Reranker for enterprise deployment | Elastic Open Inference API blog, AWS Marketplace listing | Search reranking used by enterprise search platforms |