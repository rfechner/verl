### TL;DR

This experiment is a hard negative test on correlation between in distribution delta-of-logprobs (from here on out simply called deltas) and out-of-distribution deltas. Additionally, we're correlating OOD-deltas and prevalence of behaviors. We're trying to increase certainty that the methods we're using to identify relative increase or decrease in behaviors do measure a meaningful effect.

### Experiment Set Ariadne

This experiment is about correlating deltas of out-of-distribution deltas and in-distribution deltas. The OOD samples are tuples (correct response, error augmented response), the ID samples are tuples (correct response, incorrect response with calc error).
The ID samples are per-checkpoint, we're evaluating all checkpoints on all samples (it's just easier for me to run the experiment this way... im lazy). Later on, we'll calculate the log-probabilities for (gt, aug) and (gt, err) and try to correlate along the checkpoints.
Formally, we're intersted about the correlation 

$\text{corr}(\Delta_{\text{ID}}, \Delta_{\text{OOD}})$

where $\Delta_{\text{ID}} = \{\Delta^0_{\text{ID}}, \Delta^{10}_{\text{ID}}, ..., \Delta^{80}_{\text{ID}}\}$. 

The deltas are computed as mean over per-query difference between augmented response $\hat y$ and anchor response $y$.

$$\quad \Delta^i_{\text{ID}} = \frac{1}{N}\sum_j^N \log p(\hat y_{ij}|x_{ij}) - \log p(y_{ij}|x_{ij})$$

What we'd like to observe: Decreasing likelihood delta for both sets of traces across training, a perfect correlation.
What would this result mean for my experiments? It would mean that I've shown empirically that for a single model, a single data type and a single augmentation (computation error) deltas of likelihoods for ID and OOD data correlate. It would give my previous experiments (and the ones I'm about to schedule) more weight.

What I'm likely to observe: Weak correlation or no correlation at all. This would mean that there is an effect present in either data which makes correlation hard. This could be a preference shift, pushing the OOD samples into regions where spurios effects dominate the likelihood of the trace.

What I'd be surprised to observe: Strong negative correlation. In this case, I'd have to look at the trends of the likelihoods themselves, not only the differences.

### About the experiment and the data preprocessing

At the core this experiment is about increasing trust in a method for measuring change in behaviour. At the core, we would like to measure change in behaviour but can't because it's prohibitively expensive to sample millions of traces and hard to classify them onto a taxonomy. If we instead could measure likelihoods of carefully constructed out-of-distribution samples and be certain that what we measure translates to behaviour observed in the ID model outputs, we could construct arbitrary behaviour-augmented traces (one may think of augmenting ground truths with certain reasoning strategies like divide and conquer for example) and measure the change relative to ground truths.

The issue is that by measuring likelihoods of static distributions over a set of checkpoints, we are measuring all kinds of effects - RLVR generally sharpens the models distribution, there is a preference shift towards a "problem solving output" distribution. This experiment correlates ID with OOD deltas, performing a negative test: If we see a negative correlation we can be sure that the method in itself is problematic. However: Seeing a positive correlation doesn't automatically mean the method is correct - there are still confounds even though I've given thought on how to alleviate them.

To this end, let me walk through the preprocessing steps to construct the ID and OOD pairs $(y, x)$.

#### ID samples

From a previous experiment (exp05), I've stored checkpoints of a Qwen2.5-7B model trained with GSPO on Dapo17k. The training was stable and performance as expected. From the validation rollouts of every checkpoint $i \in \{0, 10, 20, ..., 80\}$, where $i=0$ denotes the base model, I've taken the math500 rollouts to obtain 256 answers per question.

From these questions, I've built a subset of questions which have positive and (at least 3) negative rewarded traces in their set of answers. Note, that the set of questions and answers is chosen independently per checkpoint to maximize the space of possible candidates. From this subset, I've sampled 3 negative traces for further inspection. For these negative traces, I have to decide whether or not they contain a calculation error.

I've designed and iterated on a prompt for the classification process. I've settled on using Qwen3-8B for classification quite early, as i saw promising results and impressive reasoning capability. To evaluate a prompt, I've ran the experiments and chose 20 samples with a fixed seed, s.t. i may judge the model's outputs for a fixed set of diverse negative responses. I've chosen precision as the dominant metric for evaluation, as my plan is to filter for positive samples later on and I'd rather "loose" some positive samples (traces containing calculation errors) than muddy my final evaluation set. After a few iterations I've settled on a prompt which achieved 80% precision, which I deemed enough for this experiment.

From there on, I've collected and filtered the negative responses flagged as "calculation error" and mapped them back to their initial question. As the set of questions I've chosen also contained ID-positive samples, I could build pairs $(\text{correct response}, \text{false response containing calculation error})$ for the delta analysis.

#### OOD samples

To collect the OOD-samples I followed the standard recipe I plan to apply over model, data and algorithm dimensions later on. I've collected correct answers from the base model (checkpoint 0). I designed a strict augmentation prompt which fro a given ground truth $y$ lets me parse out a syntactically similar but error-augmented trace $\hat y$. I've judged the quality of generation and found the generated samples to be 1) of high quality and 2) to be close to their given ground truth in style. For example, a ground truth which contained a "divide and conquer" strategy where the model applied multiple steps, each pre-pended by a `#### Step X: Step Description` would result in the augmentation model replicating this specific form, but also inject a calculation error.

A previous problem was the occurence of artifacts within the augmented traces, where the augmented response contained reflections like "... (which is a calculation error) ..." or "... I (incorrectly) assume that ..." which I had to specifically avoid by mentioning this pattern in the prompt.

### Results

We observe a negative correlation. This is in part explained by strong preference shift over the course of training. See `/u/rfechner/verl/workspace/plots/exp12_ariadne_viz.ipynb` for these results.

I've continued by looking at general prevalence of errors throughout training, finding that the share of calculation errors of errors made on solvable questions (questions with score variance unequal to zero) decreases over the course of training - indicating a positive training effect. This lead me to hypothesize that the trend we're seeing in the ID-differences is largely dominated by the preference shift induced by RLVR and subtle trends are lost. We'd need to correct for the preference shift in order to gain a clear image - this is beyond my reach.

I've settled on continuing my analysis by trying to reduce this confound - instead of sampling the to-be-augmented samples only from checkpoint 0, we'll sample them from all checkpoints instead, keeping track of their respective step to later map back and gain a more differenced per-checkpoint view of the delta-plots.

**Correlation of error prevalence and per-checkpoint-augmentation is positive, indicating that we're measuring something meaningful**

To check this finding, I've begun effors to repeat the experiment for another augmentation - validation. It is expected that a behavior like validation increases prevalence during training. (Need to check this first...) In case the method works, we should see positive correlation with prevalence AND a increasing delta plot.

However: Measuring the ID-deltas as I've done before is skipped, as we've seen in the previous experiment that the difference is strongly confounded by preference shift, making a correlation very hard.