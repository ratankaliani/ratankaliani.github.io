+++
title = "Evo 2: Arc’s DNA Foundation Model Explained"
description = "An in-depth look at Evo 2, Arc Institute’s DNA foundation model—its architecture, capabilities, and implications for genomics."
date = 2025-06-03
updated = 2025-06-06

[taxonomies]
categories = ["bio"]
tags = ["bio", "ml"]

[extra]
lang = "en"
toc = true
comment = false
copy = true
outdate_alert = true
outdate_alert_days = 120
math = false
mermaid = false
featured = false
reaction = false
+++

In February, [Arc Institute](https://arcinstitute.org/) released [Evo 2](https://arcinstitute.org/tools/evo), a DNA foundation model trained on genomes from all domains of life. According to Arc, this represents the largest training effort in computational biology to date, with novel architectural advances to handle genomic sequences.

Over the past few decades, synthetic biologists have dreamed of programming cells as easily as we program computers: creating bacteria that clean up pollution, crops that thrive in extreme environments, or therapies that target disease at the genetic level. Traditionally, this has required painstaking manual design and years of trial and error. Evo 2 represents a potential leap forward: what if we could generate entirely new genomes [in silico](https://en.wikipedia.org/wiki/In_silico), rapidly and at scale?

Over the past few days, I got curious about Evo 2, initially because of the application of a non-Transformer architecture and the nuances of modeling on genomic data. Quickly, I found myself diving through Wikipedia and Deep Research reports to understand the biological concepts and the technical details of Evo 2. This blog post is an attempt to distill my understanding of Evo 2 into a single post. My goal is by the end you'll understand (1) why genomic sequence modeling can't simply rely on off-the-shelf Transformers, (2) how Evo 2 can be useful in today's biological workflows and (3) how Evo 2 and similar genomic foundation models may evolve in the future.

# What is Evo 2?

![Evo 2 Cover](/evo2/cover.png)

Unlike modern LLMs, which are optimized for the constraints of natural language, Evo 2 is tailored to genomic sequences. Genomic modeling is different domain than language modeling. From a low-entropy vocabulary to complex long-range token dependencies between tokens that are up to 1M tokens apart and single-token precision for clinical accuracy, Evo 2’s constraints are quite specific to genomes.

Evo 2 is a successor to [Evo](https://arcinstitute.org/news/blog/evo), which was the first large genomic language model released by Arc back in early 2024. Evo showed that large, long-context genomic DNA foundation models could viably generate “realistic” genomic sequences. But, Evo only focused on prokaryotes, and didn’t generalize well to human genomes. To accurately predict animal and human genomic sequences, Evo 2 modifies both the training data set to include eukaryotic genomes and the architecture of Evo to work for the larger eukaryotic genomes.

Evo 2, according to Arc, was trained on over 2,000 H100 GPUs for several months, putting the estimated cost of the run at ~$10M. With Evo 2, you can:

- Generate novel genetic sequences across both eukaryotes and prokaryotes that exhibit high fitness.
- Predict the functional impact of specific genetic variants, such as BRCA1 mutations linked to increased breast cancer risk.
- Improve sequence‐fitness predictions at inference time: giving the model more compute resources during inference leads to better genomic‐fitness estimates, similar to how test‐time scaling boosts reasoning performance in LLMs.
- Handle up to 1 million base pairs of context when generating sequences.
- Leverage mechanistic interpretability to see what Evo 2 “learned” from raw genomic data.

These capabilities are impressive, but how does Evo 2 *really* work? Sure, by increasing the dataset size, model size or compute, [you expect language models to get better](https://arxiv.org/abs/2001.08361). What makes Evo 2 specifically suited for genomics?

# Quick Primer

To make sure that you don’t get overwhelmed by the breadth of biological knowledge in Evo 2 and spend several hours asking o3 questions (like I did), I’ve distilled some basic biology concepts relevant for understanding Evo 2. These include both the data that Evo 2 is modeled on, and some terms for the tasks that Evo 2 is applied to. If you're already familiar with these concepts, feel free to skip to the [Training & Architecture](#training-architecture) section.

## Genome Modeling

**Nucleotide Sequences**: Evo 2 models DNA sequences. These linear sequences are composed of nucleotides that encode genetic information: adenine (A), guanine (G), cytosine (C), and thymine (T). According to the paper, Evo 2 also pre-trains on RNA sequences, and RNA sequences use Uracil (U) instead of Thymine (T). In DNA, this is abbreviated as “AGCT”, and in “RNA” as “AUCG”. and I’ll refer to them as nucleotide or genomic sequences throughout the rest of the blog post.

**DNA → RNA → Protein:** Evo 2 explicitly models DNA sequences and RNA sequences because it’s pre-trained on them. *But how does it model protein sequences?* DNA and RNA contain the instructions for protein synthesis, so Evo 2 can learn statistical and structural features of DNA/RNA that implicitly encode proteins. Evo 2 learns which RNA transcripts are likely to be stable, expressed and translated into functional proteins. When I cover mechanistic interpretability with Evo 2, I’ll show specifically how it has internalized these features. This is known as the “[central dogma](https://en.wikipedia.org/wiki/Central_dogma_of_molecular_biology)”, with the caveat that RNA does more than just implicitly encode proteins ([obligatory XKCD #3056](https://xkcd.com/3056/)).

**Variant-Effect Prediction:** Computational bridge from raw mutation to clinical decision, based on a genetic mutation you can determine if a specific condition is more likely.

**SNVs:** Single nucleotide variant - a change in a single base in the genome sequence. Most are benign and some are harmful.

**Variant Pathogenicity:** How likely a genetic variant (usually an SNV) is to cause disease. Falls into 3 categories: pathogenic (causes disease), uncertain significance and benign.

## BioML

To contextualize Evo 2, it’s also important to understand the lineage of models that it follows. There are two main model lineages that are relevant to Evo 2, large protein language models, trained on protein sequences, and large genomics models, trained on genomic DNA sequences.

[AlphaFold](https://alphafold.ebi.ac.uk/), one of the first large protein language models (PLMs) came out in 2020 and predicted protein structures. It clearly demonstrated how ML algorithms can “learn” biological structures better than humans for a laboratory relevant tasks. Then in 2021, [ESMFold](https://github.com/facebookresearch/esm) demonstrated how the transformer architecture could be applied to protein sequences to predict structural features purely from the embeddings of large-scale PLMs. 

Around 2021, most genomics modeling was highly task-specific. Models such as [Enformer](https://deepmind.google/discover/blog/predicting-gene-expression-with-ai/) and [GenSLM](https://github.com/ramanathanlab/genslm) were used for epigenomic signals and microbial genome generation, but didn’t generalize well beyond their training set. In 2024, Evo marked a similar transition to ESMFold, but for genomics data. Evo was the first model to show that a long-context genomic sequence model trained on generic prokaryotic data could yield sequences with high prokaryotic sequence fitness *without any task-specific fine-tuning*. Evo 2 naturally followed Evo, but with architectural changes to perform well for eukaryotic data.

![Evo 2 Model Size](/evo2/evo2_model_size_comp.png)

# Training & Architecture

With this background on the biology and model lineages behind Evo 2, let's dive into what's new and unique about the Evo 2 architecture.

Evo 2 uses a [“multi-hybrid” architecture](https://arxiv.org/abs/2503.01868), a Transformer-like architecture that extends the original Evo architecture for better efficiency and longer context. In this section, I'll go over the motivations for Evo 2's architecture (StripedHyena 2), the architecture itself, and the training recipe that Evo 2 used.

## Motivations

The motivations for Evo 2’s architecture are quite different than those for traditional language models.

1. The context length required for effective genomic modeling is on the order of millions of base pairs due to long-range dependencies between nucleotides. In large nucleotide sequences, “far-apart” bases can still interact with each other. Nucleotide strands fold in 3-D and folding will pack distant nucleotides side by side. In fact, SNVs can cause conditions such as thumb duplication.
2. The vocabulary size for Evo 2 is O(10), with most tokens concentrated in the nucleotides for DNA, AGCT, and RNA, AGCU. This is drastically smaller than the vocabulary size in [modern LLMs, which are typically between 10K to 100K tokens](https://github.com/openai/tiktoken).
3. Single-nucleotide resolution is critical. To make clinically relevant predictions for SNV’s (among other targets), you need to know exactly which nucleotide changes. Increasing the size of the vocabulary to reduce the context length is a patchwork solution and is untenable for this reason.

Traditional transformers which are used for language modeling are not able to effectively handle 1M tokens of context with global long-range dependencies. Most LLMs (e.g. GPT-4, Claude4) max out at a 200K context window due to the [quadratic compute requirement for attention](https://arxiv.org/pdf/2307.08691). 

Methods of achieving longer-context with transformers such as local + periodic global attention still don’t reduce this compute budget requirement. So, out-of-the-box transformers will be quite expensive for modeling genomic sequences at the scale of millions of tokens.

## StripedHyena 2 Architecture

Before explaining Evo 2’s architecture Striped Hyena 2 (SH2), I’ll explain Striped Hyena 1 (SH1), the architecture for Evo.

SH1 combines the [Hyena state-space convolution](https://arxiv.org/abs/2302.10866) with self-attention layers. SH1 is similar to [Mamba](https://arxiv.org/abs/2312.00752), except that SH1 fuses the convolutional operators and attention layers to push more computational efficiency and tighter coupling of local & global features. At a high level, Mamba’s simple sequential convolution → attention blocks can fit into existing Transformer pipelines, whereas SH1 requires custom fusion logic. Because SH1 is applied to genomic data, where the context length is much longer than in language modeling, this tradeoff makes more sense. Fusing together the attention and convolution blocks amortizes the overhead to reduce inference and training compute, which yields 2x speedup over dense transformers. 

Mamba and SH1 are known as “hybrid” architectures because they combine two key components: attention and state-space convolutions.

Evo 2’s architecture, called SH2, builds on this idea by using a mix of attention layers and three types of convolutional operators—short, medium, and long. Each operator is specialized: short convolutions capture local (nearby) dependencies, medium convolutions handle patterns over hundreds of tokens, and long convolutions capture relationships across very long stretches of the sequence. This combination is why SH2 is referred to as a “multi-hybrid” architecture.

A major advantage of SH2 over SH1 is its use of multiple convolutional layers, rather than just Hyena-LI, which increases training and inference speed.  Specifically:
- **Hyena-SE (Short Explicit filters):** Focuses on recalling information from nearby tokens (local context).
- **Hyena-MR (Medium Regularized filters):** Handles dependencies over several hundred tokens, maintaining efficiency and performance.
- **Hyena-LI (Long Implicit filters):** Captures long-range dependencies across the sequence.

The paper draws a nice parallel between the new convolutional operators and the classic attention operator.

> Hyena-MR is to Hyena-LI what sliding window attention is to the classic attention operator. 

By leveraging more efficient convolutional operators for short, local recall, SH2 gets a 2x-3x speedup in training over SH1. SH2 only pays for long-range context occasionally, rather than in each layer.

![Evo 2 Architecture](/evo2/evo1_evo2_architecture.png)

Below, you can see the scaling results for a dense transformer against SH1 and SH2. At longer contexts, you can clearly see that the Hyena-SE and Hyena-MR convolutional kernels out-perform MHA (multi-head attention) as you’re no longer paying the quadratic overhead of attention. The use of Hyena-SE and Hyena-MR kernels increases training and inference throughput!

![Evo 2 Scaling](/evo2/training_speedup_vs_transformer.png)

More details on how the convolutional kernels are implemented + further results can be found in the [Evo 2 paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) and the [sister ML paper](https://arcinstitute.org/manuscripts/Evo2-ML).

## Dataset

Evo 2’s training data (called [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2)) comprises over 9T DNA base pairs spanning all domains of life (bacteria, archaea, eukarya). 

To capture both local functional elements and long‑range genomic dependencies, two training phases were employed: an initial “short‑context” pretraining on 8,192‑bp windows enriched for genic regions, followed by a “midtraining” stage that extended context to 1M base pairs, with sequence composition shifted toward whole‑genome samples. By combining these two training phases, Evo 2 achieves single‑nucleotide resolution and the diversity needed to generalize across everything from mitochondrial micro‑genomes to complex eukaryotic chromosomes.

![Evo 2 Training Composition](/evo2/training_composition_mix.png)

As seen in the charts below, the vast majority of FLOPS are applied in pre-training. From pre-training the model gathers knowledge about biological structure, and mid-training extends this structure from just eukaryotic genes to genomic sequences from the genome data bank.

![Evo 2 Training Tokens](/evo2/tokens_consumed_training.png)

Now that you have a basic understanding of the training data and architecture of Evo 2, let's dive into how to use Evo 2 to generate novel genomic sequences!

# Sequence Generation with Evo 2

Evo 2 can generate new DNA sequences by starting from a short input sequence and predicting what comes next, one step at a time. In this section, I’ll show you how to use Evo 2 to either continue a DNA sequence with no extra guidance, or to search for a sequence that meets a specific goal.

## Zero-Shot Novel Genomic Sequences

The API for requesting a genomic sequence from Evo 2 is quite simple: pass the input and the number of additional tokens and the model will generate the corresponding nucleotides after the initial sequence.

```python
dna_seq = evo2_model.sample(prompt="ATGCTG", ..., number_of_tokens=1000)
```

To demonstrate how Evo 2 generates “novel” genomic sequences and generalizes well, the Evo 2 team generated a diverse set of “viable” eukaryotic genomes starting from a human seed. While there were no wet-lab experiments employed to validate the generated nucleotide sequences, the Evo 2 team employs computational methods to assess that the generated sequences are “similar” and contain the critical components for mitochondrial DNA. Specifically, they deploy [BLAST](https://en.wikipedia.org/wiki/BLAST_(biotechnology)) analysis to verify sequence homology of the generated sequence against different naturally occurring organisms and [AlphaFold 3](https://alphafold.ebi.ac.uk/) to validate the generated structures matched expected mitochondrial protein complex folds. In simpler terms, scientists check that both the raw “sequence” and the proteins that it encodes are “viable”. 

Below, is a sample of how the Evo 2 model was used to generate a set of genomes from a small human mitochondrial DNA seed. The created genomes ranged from being most similar to a sheep or a fish, all from the same seed.

![Diverse Genomes from Human Base](/evo2/diverse_genome.png)

## Scaling Test-Time Compute

Unlike traditional LLMs which can be prompted with additional language context to steer the output of generation, Evo 2 can only process nucleotide sequences as input. This is fine if you just want to get *a* nucleotide sequence which looks *reasonably* natural. But *what if you want to get one with specific characteristics -* say a sequence that’s more likely to bind to a protein target? Because Evo 2 only parses nucleotide sequences, once you’ve generated a genome, there’s no way to guide the model in a specific direction with natural language.

*So, how do you do you do directed search for nucleotide sequences that have specific conditions beyond “natural viability” at inference-time?* 

For this, we can look to modern ML, where scaling test-time compute has become the new hot thing. At a high level, techniques for [scaling test-time compute](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) can be broadly categorized into **self-refinement** and **searching against a verifier**. 

Because there are many heuristics (known and unknown) that you may want to optimize Evo 2 for within genomic sequence generation, the Arc team chose searching against a verifier**.** Specifically, the Evo 2 team applies beam search to select the best option at each step, which is illustrated below. In beam search, you pick a heuristic function to evaluate each step of generation, sample N generations and keep the K top samples according to the heuristic.

![Beam Search](/evo2/beam_search.gif)

Let’s walk through how the Evo 2 team uses beam search to generate proteins for a specific heuristic: *chromatin accessibility* patterns. Chromatin accessibility refers to how “open” or “closed” regions of DNA are. Evo 2 uses Enformer and Borzoi as heuristic functions for evaluating a generated nucleotide sequence. Both models yield whether a nucleotide will have accessible chromatin given a window of nucleotides around the target. With beam search, Evo 2 guides the generated genomic sequence to encode specific chromatin accessibility patterns in nucleotides with the following protocol:

1. Sample N times (fan-out) from Evo 2 given the same prompt.
2. Score the N samples with Enformer and Borzoi.
3. Select the top K chunks based on the and append to the prompt.
4. Repeat until you’ve generated a nucleotide sequence of length L.

With this approach, the Evo 2 team was able to generate genomic sequences that encode chromatin accessibility patterns that match the Morse code encoding of “Evo 2” and “ARC”. You can see these encoded in the chromatin accessibility diagrams below.

![Chromatin accessibility](/evo2/chromatin_message.png)

# Evo 2 + Mechanistic Interpretability

The Evo 2 team worked closely with Goodfire to train sparse auto-encoders (SAEs) on latest representations from Evo 2. SAEs trained on Evo 2 uncover several biologically relevant features, purely from nucleotide sequences:

- Canonical gene structures (CDS, UTRs, exons).
- Structural motifs (α-helices, RNA stem-loops).
- Protein structure characteristics

Although all of the identified features above from Evo 2 are biological features which are commonplace in literature, there *could* be learned representations of “unknown” biological features in scientific literature discovered by the mechanistic interpretability. If large models can learn “deeper representations” beyond what’s known in literature, just training a model and interpreting it could be enough to unlock new research.

## Visualizing Evo 2's Features

I really enjoyed playing around with Goodfire’s [mechanistic interpretability visualizer for Evo 2](https://arcinstitute.org/tools/evo/evo-mech-interp) because it concretizes what the utility of feature annotation is. The SAE trained by Goodfire can identify a semantic concepts in a nucleotide sequence you provide. Their interface shows activations on the supplied nucleotide sequence for several features in the model, such as features that fire for nucleotide sequences encoding proteins for α-helices and $\beta$-sheets.

![Mechanistic Interpretability Visualizer](/evo2/mech_interp_visualizer.png)

Using AlphaFold3, you can model the 3D protein structure for a nucleotide sequence. You can see in the image below that the the positional pattern of the features matches the simulated secondary-structure elements of the encoded “protein”. This is one of the cooler representations of how genetic language models “learn” 3-D structure of proteins, without being explicitly trained on it.

![Annotated protein structure](/evo2/protein_structure_annotated.png)

## Steering Genomic Generation

Okay, so now we know how to do directed search for a specific heuristic function, but what if there isn’t a well-defined characteristic? What if we just want to explore the space around a generated genomic sequence to see if we can get a sequence that fits our constraints?

From [Goodfire’s blog on Evo 2](https://www.goodfire.ai/blog/interpreting-evo-2):

> Unlike language models that process human-readable text, these neural networks operate on DNA sequences—a biological code that even human experts struggle to directly read and understand…

> The potential impact of steering Evo 2 is particularly significant: while language models can be prompted to achieve desired behaviors, a model that 'only speaks nucleotide' cannot. Learning to steer through features would unlock entirely new capabilities.


In biology, there is no equivalent to prompt engineering as the model only understands nucleotide sequences. The decision space for guiding outputs is limited to directed search (beam search), training models to seek specific characteristics with rewards (RL) or steering. Of the three, steering is the only category that does not require test-time compute to scale. Rather, steering can be used to directly guide sequences towards having characteristics in the latent space of DNA without requiring an explicit heuristic function.

# Future of Evo 2

Now that you have an understanding of Evo 2 and how to use it, what does this all mean for the future of synthetic biology?

Reading the socratic dialogue in [these](https://www.owlposting.com/p/a-socratic-dialogue-over-the-utility) [blogs](https://www.owlposting.com/p/a-socratic-dialogue-over-the-utility-a78) by [owlposting](https://x.com/owl_posting) about Evo 2 was helpful for me to grok what new science Evo 2 unlocks. At a high level, he argues that we need real-world validation of Evo 2's DNA variant pathogenicity prediction to use it in production workflows and that large-scale synthetic genome generation is cool, but bottlenecked by real-world DNA synthesis costs today. Evo 2 was only evaluated with digital models, and not on any real-world biological experiments. As such, reproducing the utility of Evo 2 in real-world experiments will be necessary and will take time and failures before the model is used in production.

In terms of future improvements to Evo 2, I’m particularly excited about **steering genomic generation with advanced interpretability.** Having an interface that makes steering DNA sequences [as seamless as Goodfire's interface for steering a language model](https://platform.goodfire.ai/chat/new?model=70b) will allow designers of novel genomic sequences to guide their generations more effectively. Additionally, I expect to see **more heuristic functions beyond chromatin accessibility used for genomic generation with Evo 2**. Directed search towards characteristics with a well-defined heuristic function will be better served by techniques like beam search, rather than steering, so expect to see researchers apply inference-time scaling on Evo 2 towards characteristics beyond chromatin accessibility. One note on interpretability is that biology is likely under-explored for useful semantic concepts because the number of people that can use DNA foundation models is **significantly less** than the number of researchers that would benefit from language models. As such, even though Evo 2 has been trained on a wide set of genomic data, extracting useful “novel” semantic features is difficult, because the semantic features in biology require scientific discovery to be found.

LLMs today are more useful for real-world tasks because a reward-model for “language” *can already be* *approximated for* *a reasonable cost*. Empirically, this has been borne out through RLHF for qualitative reasoning tasks or RL reward models for constrained tasks such as math and coding. On the other hand, DNA foundation models like Evo 2 don’t have the same high fidelity data. Getting reward signals on biology is difficult because you need to run experiments *over time* in the real world, which is both more costly and hardware-intensive than getting reward signals when training LLMs. RL environments for LLMs don’t need to interface with the physical world for the most part, whereas for DNA foundation models, connecting high-throughput real-world experiments to the RL environment will likely be necessary.

For Evo 2 and models like it to truly transform biological research, we’ll need better virtual environments for training, tighter integration with high-throughput experiments, and smarter reward modeling that bridges the computational-to-experimental gap. Biology is complex and expensive to iterate on, but Evo 2 is a great example of how much nuance there is in applying modern large models to a new domain, as well as what the true bottlenecks in making the models useful to everyone.

*Thanks to [Chris Zou](https://x.com/chriswzou) and [Darya Kaviani](https://x.com/daryakaviani) for their feedback & support on this post! I appreciated Asimov's [post on Evo 2](https://www.asimov.press/p/evo-2) and owlposting's [socratic dialogues](https://www.owlposting.com/p/a-socratic-dialogue-over-the-utility), which were both an inspiration for this post.*