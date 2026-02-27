#### 07.02

- Re-did pass@k plots however: Due to lowered variance in the estimate of Qwen2.5-7b base with nsamples > 2048, pass@256 estimate isn't binary any more.
	-> We currently fix this by rounding the pass@256 values for the UpSet plots to binary values.
	-> This also has another impact: Pass@k Training is now the strongest exploration method, KL-Cov prevents forgetting.
	
#### 06.02

- check whether exp17_more_rollouts has finished.
	- redo the comparison plots.
	- crahed due to inability to acquire GIL.
- check whether klcov_passk training run has finished
	- finished to checkpoint 60.
- check whether different seeds for kl-cov have finished rollouts
	- make the UPSet Plots for different seeds
 
- Should re-do the pass@k plots once rollouts are finished.
#### 05.02
- different seeds have finished, now just need to run rollouts for seed2 and seed3.
	-> should do this only for checkpoint 30.
	-> also: run the rollouts for klcov_passk.

- data driven has finished. Replot datadriven.
- judging has finished. Check how long the judging took.
	- should append the hard negative case to the prompt: Python is not verification. 
	- add reward to the prompt dataset, remove old_index.
	- When the solverate of the trained rollouts is much higher, the chance of seeing a verification increase.
	I should think about importance sampling techniques to remove this confounder.	

#### 04.02
- check data driven grpo finished
- restart judging, but on 4GPU machine. Should get an estimate of how long this takes and whether i have to split this into smaller jobs
	- subsampled to 16 rollouts per question and restarted juding procedure.

- check training dynamics, new outcome based metrics.

#### 03.02
- wait for rollouts to finish, -> subsample_exp13_for_data_driven_analysis.ipynb -> start verl/workspace/exp13_grpo_data_driven.sh
- check logprob rollouts have finished
- check juding has finished

#### 02.02

- rerun grpo experiments, as gspo is degenerating across models
- check whether all models have upto checkpoint 80, further work:
	- rollouts
	- re-plotting für outcome based methoden
	- ich denke für die anderen methoden wird es genügen GSPO zu vergleichen.

#### 14.01
- thesis:
	- delta correlations, sariadne

- experiments:
	- data driven experiments -> subsample per question.

#### 13.01

- data_driven experiments start now, but it seems calculating the logprobs
	takes too long. I should think about further reducing the
	load.
	Current load: 1 checkpoint: ~8 * 256 * 630 = 1290240 "rollouts".
	Have to reduce this. Could I instead make a subdirectory "exp13_rollouts_reduced", where
	I've subsampled the data? ~16 sampled per question should suffice.

#### 11.01

only writing day. Related works or describing algorthims used.
- check on data-driven experiemnts.

#### 10.01

- Connect research questions with results.
- Write section on research questions
- get data driven experiments to run
	- refresh token
	- make sure huggingface reads the exported token for login.
- paste first placeholder images to the respective secion
- develop a workflow of saving images & synchronizing them through github.
- start subsection of ariadne experiments under the logprob results section
 
#### 07.01

- data driven experimente scheinen alle fehlgeschlagen zu sein
	- torch dynamo compiler weirdness. Should migrate back to flashinfer, also probably not start the jobs the same time.
	
#### 31.12

- Trainings fertig?
	- Alle trainings bis auf KL-Cov experimente bis Scheckpoint 80 gekommen.
	- qwen2.5-1.5b kl-cov nicht gestartet, bzw nur bis step 1 gekommen, da nicht genug nonzerostd gruppen für einen Batch gesammelt werden können.
	- plot training dynamics
	- versichern, dass alle methoden wie erwartet gelaufen sind.
	=> Methoden schauen "normal" aus. Es gibt einige ausreißer aber im Großen Ganzen beobachte ich, was zu erwarten war.
- Rollout experimente planen
	- Muss die checkpoints trimmen 
		-> alles über checkpoint 80 löschen
	=> checkpoints > 80 renamed.
	=> rollouts started
- Lesen: Differential Reward Smoothing paper
- Schreiben:
	- Ich dachte da an einen Methodenpart für die Ariadne Versuche.

#### 30.12

Ich sollte die Versuche von Projekt Ariadne sortieren und vereinheitlichen. Es kann nicht sein, dass ich 6 verschiedene notebooks herumfliegen habe, welche
dieselben FUktionen implementieren oder Datensätze neu laden, anstatt diese from-disk zu laden. Außerdem sollte ich mir mehr GEdanken über das mapping von
augmentationen oder id-flagged antworten auf die ursprünglichen Fragen/Antworten machen.

- Ich habe ein wenig mehr Ordnung in das Durcheinander gebracht. Habe jetzt nur den OOD-delta logprob job gestarted, da ID-delta analyse
schwer ist (die Präferenzverzerrung beeinflusst jegliche Interpretation der Resultate).
- Die plots sollten morgen gemacht werden. Ich habe allerdings schoneinmal vorgegriffen und den Anteil der Vorkommnisse von Validationen
innerhalb der getesteten Antworten errechnet und komme auf ein "gutes" Ergebnis: der Anteil steigt stetig.
- Das einzige REsultat welches ich noch benötige um meinen log-prob versuchen Standfestigkeit zu geben ist eine positive Korrelation mit den OOD-logprob deltas.
=> Logprob resultate schlecht bis gemischt: Es gibt eine negative korrelation, allerdings sollte ich nicth das Kinde mit dem Bad ausschütten.
Das wichtigste ist, dass der Versuch nun durchgeführt wurde und ich mir sicherer bin, dass die Ergebnisse der Delta plots stark von confoundern
beeinflusst sind. Ich muss morgen Klarheit in die Resultate bringen und mehr über die Implikationen nachdenken.

#### 29.12

- The judging of whether or not a correct solution contains a verification seems to work. I'm now able to generate the id-delta-verify dataset.
- The OOD-verfy dataset was already created, however: Instead of verification only at the last step, I've prompted the model to include
	verifications of every intermediate step. This may induce a confound into later analysis. I think i should prompt the model to only 
	include a verification step before submitting the final answer.

#### 18.12

- Ariadne per-checkpoint errors und validations nocheinmal laufen lassen -> 3 neue jobs.
- Errors per-checkpoint OOD:
	- neues delta-ds.jsonl generieren
	- neuen job laufen lassen, welcher die logprobs für die ood-checkpoints per-checkpoint berechnet
	- job started
	
- Validation:
	- id-deltas-verify.jsonl und ood-deltas-per-checkpoint-verify.jsonl generieren.
	- sollte zuerst einmal die qualität des outputs beurteilen.
	- id-outputs waren auf fehlern evaluiert. Neuer generations-job gestarted
	- ood-outputs-per-checkpoint-verify job started.

- Vorbereitung auf das Starten der Training jobs.
	- Kann ich Versuche auf OLMO3/EuroLLM/SwissLLM machen?
	
#### 17.12

- Ich sollte die letzten Tage vor Weihnachten noch einmal überlegen, wie ich mir meine restliche Zeit einteilen möchte. Thaddäus gibt mir einen
	richtigen Hinweis: Ich sollte mindestens die hälfte der restlichen Zeit welche ich für Experimente eingeplant habe noch für die Vervollständigung der
	experimente auf allen ebenen, (Modell x Algorithmus) planen und schedulen.

	15.02 Abgabe
	08.02 Abgabe an Thaddäus
	01.02 Polishing
	08.01 Schreiben
	31.01 Weite Experimente starten + plotting vereinheitlichen
	28.12 Logprobs experimente wdh und graben

- Ich sollte organisatorisch feststellen:
	- Thesis Registrierung
	- Verteidigung Termin fix
	- AI Statement

- Tasks für den restlichen Dezember:
	- Ariadne Augmentationen wdh, dieses mal mit samples per-checkpoint.
	- Delta Plots wdh. mit neuer Ariadne Augmentation
	- Delta Plots mit OOD/ID suffix analyse. (stretch goal)

- Ariadne sollte für per-checkpoint augmentation und ein behaviour wie z.B. Validation wiederholt werden.

- Experimente Enumerieren -> Arbeitsschritte ableiten
	- drawio


#### 15.12

- Der Hauptkritikpunkt den ich an meiner Forschung habe ist, dass ich keine klare Frage beantworte. Die experimente die ich habe
	sind interessant und werfen weitere Fragen auf, sie stellen aber keine Aussagen fest. Der Fokus muss also auf der Findung einer
	soliden Forschungsfrage liegen, von der aus ich dann Experimente ableite.

- Ich habe bereits Forschungsfragen gesammelt, diese jedoch nicht festgeschrieben. Das letzte halbe Jahr habe ich viele Ansätze verfolgt,
	in der Hoffnung einen "Glückstreffer" zu erlangen - eine Beobachtung aufgrund welcher ich dann eine "Neuigkeit" in meiner Arbeit hätte.
	Dieses letzte halbe Jahr war eine Erkundungsphase und eine Orientierungsphase, jedoch sind keine Interessanten Fragen gestellt worden - eher
	wurde erwartet, dass sich diese mit der Zeit ergeben würden.

- Es war immer die Absicht, eine Studie zur Erkundung während des Trainings von Sprachmodellen zu machen, jedoch ist der Begriff der Erkundung und das Feld selber
	sehr neu und somit sind die Begriffe noch nicht gereift. Es gibt keine Definition von was Erkundung überhaupt sein soll, eher wird ein Rückschluss gezogen,
	dass wenn ein modell eine hohe Punktzahl erreicht, es den Raum der Antworten wohl erkundet haben muss, da ja ein Delta in der Lösungsrate zu sehen ist.

- Ich denke der Strom meiner Experimente läuft in die richtige Richtung - es ist nur so, dass ich keine konkreten Fragen distilliert habe, welche ich beantworten könnte.
	Es wird mir nichts weiteres übrig bleiben, als einfach Fragen zu sammeln und diese dann stück für stück weiterzuentwickeln oder zu verwerfen.
	Derzeit kann ich folgende interessante Fragen beantworten:
		- An welcher Stelle der Trace wird erkundet? Prefix/Suffix analyse.
		- Werden "Calculation Errors" durch RL-Finetuning mehr oder weniger? -> Frequenz analyse
		- Korrelieren Wahrscheinlichkeiten von Generiertem verhalten mit In-Distribution Verhalten? -> Ariadne
			  
#### 25.11
- think abt research questions.
- blueprints for kl-div

#### 24.11

`dones`
- bin by ground truth likelihood and by delta.
	- I've read through the LLM generated code and
	also binned by ground truth likelihood.
	- I more and more believe that trying to show some increase in
	behaviour is very hard for the current data. The log-prob for generating
	the ground truth behaviour decreases over training.
	- We can still say that the likelihood for seeing diverse reasoning
	behaviours approaches the likelihood for the initial ground truth, but
	that statement carries less value, the less likely the gt response is.

	- I'd have to repeat the experiments with the generated responses from
		other models to be sure about this. Would need some more time.
	- Note: From the data-driven (KL) experiments we know that the likelihood
		for generating Base-model ground truths diminishes over time.

	- What I'd need is a way of connecting the low-likelihood differences
	to high-likelihood differences. Then, I'd be able to draw a general
	conclusion. However now I fear my findings aren't telling much.
	Ideally, I'd augment per-checkpoint ground truths and look at the
	trend over the course of training. However, then I'd need to do this
	per-model-per-method-per-checkpoint which is infeasible.

	- Let's still continue, collect four "quadrants" of likelihoods and
		present on tuesday.

- lets correlate these log-likelihoods with some success metric like accuracy
- bring the likelihoods into the same plot
	- Plot the base likelihoods for the behaviours into the same plots perhaps with logarithmic axis.

`current`
- kl-cov instead of mean-likelihood plots.
		 	
`backlog`
	
#### 21.11

`dones`
- re-generate training dynamics plots
- start exp05_rollouts_qwen2.5-7B again for GSPO, DrGRPO, Entropy Reg
- look whether exp08/grpo/reasoning_types and error_types as well as exp08/kl-cov/reasoning_strategies have finished.
- exp08: repeat exp06 with qwen2.5-7b models
- exp09: repeat exp04 with qwen2.5-7b models

`current`
- towards the embedding lens: I think that the data driven and lp delta experiments are useful but:
	they don't really tell us to which extend and algorithm has explored different solution strategies throughout the
	training. Shouldn't we rather look at the set of embeddings for responses over all rollouts to judge breadth of exploration?
	Also: Looking at validation rollouts itself is perhaps not telling the full story: Shouldn't we also look at training rollouts to judge effective exploration?
	-> ponder more on this.
	  
`backlog`

#### 20.11

`dones`
- summarize the full log-prob pipeline for gpt4-gt-X-aug data
	- should i already do this in latex? couldn't hurt tbh
- i think i should start thinking about formulating relevant research questions which i am able to answer with
	my experiments.
	- i think my research gives new viewpoints on how algorithms change output distributions of models
	- exploration is a necessary condition for solving environments but it isn't clearly defined,
	rather there is an intuition on what it means for an algorithm to explore.
	There is the general intuition that solving a hard problem means the agent must've explored the space of
	solutions around the correct one.
- look at the llama-gen exp06 again. Seems to still be bugged.		
	- i think i should add a hardcoded check to the python script which determines whether the jsonl file has numlines % numgpus == 0.
- data-driven: do the 0.1-percentile idea.
- data-driven: do the k/N normalized plots and identify the relative positions k for which the delta is highest.


`current`
- correlate deltas with metrics. The delta log-likelihood comparison is only useful when
	we can correlate the results with performace, i.e. we see the likelihood for "deductive reasoning" increase and math benchmarks
	going up or the lps for errors to decrease and the perf goin up.

	- To this end, I should start/look at more methods. Most interestingly would probably be looking at
	a method which degenerated during training to see whether that reflects in the logprobs.
	- look at started experiments.
	- grpo -> reasoning types and error types havent finished
	- klcov -> reasoning strategies hasn't finished.
	

#### 19.11

`dones`
- have another close look at Dr-GRPO and GSPO. These baselines should have noticable increase in performace 
	- Lets begin with looking at the papers for Dr.GRPO and GSPO again. 

	- DR. GRPO:
		- remove sequence length normalization, remove std-normalization.
		- Lets see in code whether I'm doing these things correctly.
		- I'm correctly dropping the sequence-length normalization.
		- Also correctly not normalizing by std of per-question rewards.
	- GSPO:
		- Instead of clipping per-token likelihoods, we're clipping per-sequence.
		The likelihood of a sequence is the exp(mean(logprobs)). Based on this, we clip.
		Additionally, we're using other epsilons.
		- we're using correct epsilons (eps from paper are left: 3e-4, right: 4e-4).
		- we're using correct loss mode.
		- code seems to be implemented correctly.
		- It's hard to judge whether the implementation does it's job correctly tbh, as they do not share
		alot of hyperparameters in the paper.

	=> Started trainings for gspo, drgrpo, entropy_reg within scope of exp05.	
	Will have to look at training dynamics to judge whether there is still issue.
- reply to thaddäus notes.
- look at whether we can implement a "deep exploration" method.
	- @prasanna for hints.
	- look at literature.
	- looked at DIVER:
		- they use BLEU score matrix to assign an auxiliary diversity reward. I could probably implement this,
		seems doable.

- Further think about how to measure exploration.
	- I think another way to look at exploration is to look at occurences of "low-likelihood" tokens of rollouts
	generated by the trained model under the base model. What RL likely does is not to generate completely new
	"unlikely" responses but it reinforces islands of tokens which are unlikely under the base model, such that
	the base model wouldn't have reached the trained models response.

	Currently I'm taking the mean over the logprobs, but couldn't we also do something different? Could we look at the
	mean over the 0.1 right-tail percentile of likelihoods for the trained response under the base model? 

`current`
- I have to tie the logprob results to performance metrics like accuracy. Prasanna said to correlate accuracy with
	these deltas.
	- started delta-logprob experiments for other methods as well.
	- if I'm going to do that anyways, should i do it with the qwen2.5-7b models?
	I wanted to repeat the logprob tests anyways.
 
`backlog`

#### 18.11
- post-meeting:
	- try to identify faulty prompt by isolating batches.
	- increase log_prob num_max_tokens?
	- seems to be an issue when a batch has only one sequence.
	- just isolate the bad batch and print additional checks whether model_input ids are int or bfloat.
		
- prepare for meeting:
	- work towards "full picture" for analysis:
		- DATA DRIVEN:
			- look at log-prob of initial k tokens instead of whole trajectory
			- look at log-prob of last k tokens
			- (stretch goal: cross-method comparisons)
		- DELTA:
			- figure out the root cause of "k_partitions" being float instead of int.
			- dataset seems to be failing at a specific prompt due to weird padding logic.
				- we pass floats as model_inputs somewhere.
				- because of this, model forward call fails.
				- there was a malformed response in the dataset -> assistant content was "".
					=> when tokenizing empty response, we get float32 input_ids.
								
			- repeat logprob plots, fixing the nits from previous meeting.
				- include the llama-generated plots.
				- swap y-axis
					- => should be Delta-lower => behaviour chance decreases.
				- from the "likely" bin of each experiment:
					- make a plot where likelihoods per-behaviour are on the same y-axis for easier comparison.
				
		- overview: connect drawio to current plots and results
		- connect everything back to the original idea of measuring exploration 

#### 17.11

- thesis orga stuff
- would like to finish for tomorrow's tag-up:
	- read thaddäus responses
		- interesting analysis to back up the plot:
		Could I identify the groups of low-delta and high-delta and draw some samples from the
		datasets? This could give some evidence to what we're suspecting makes sense.
		
	- plots for likelihoods for llama-generated responses
	- possibly look at exp07 to further the logprob idea.
		- I should try and see whether the current set of experiments has run through,
		in which case I should collect the set of correctly answered questions.
		- based on the set of correctly answered questions, I should ask a strong foundation model (Qwen32B?)
		to classify, whether a behaviour augmentation makes sense in the context.
		- I'll end up with list of applicable questions for each behaviour category.


#### 14.11 + 15.11

- did some work on peers paper
- some experiments to restart/push?

#### 13.11

- exp07 -> have rollouts run through?
	- everything apart from gpt-oss has run through.
		-> gpt-oss has problems with MOdelConfig for vLLM: mxfp4 is unknown quantization.
		-> could retry and load model with fp16 instead
		
- exp03/exp06 -> can we restart with llama-augmented prompts?
	-> should be able to do that just for reasoning types and reasoning strategies. Note: i looked at the
	generated traces and they're super noisy for error types. Seems like a parsing error most of the time.
	-> Likely related to examples i provide in the context for EIC_Taxonomy.
	-> make exp04_extension shell script and start jobs.
	-> try to generate the augmentations with gpt-oss-20b as well. 
	-> 

- exp05 -> vizualizations?

#### 12.11
- analyze exp05 rollouts and make plots
- create reasoning types manual dataset?
	- discussion: which data to take?
		- should take something easy for fast iteration cycle first
		- later, we can expand to harder questions if we do not see correlations with machine generated augmentations
		
	- just pick the first five gsm questions and write augmentations for quick iteration.
	- on hold.

- Almost all of these gradeschool questions are too trivial to inject behaviours like "Pattern Recognition" into
	the answer. I believe a good question type must a) be hard enough to not be solved immediately by basic
	calculations and b) allow a diverse set of answer strategies. Only under those premises an application
	of a reasoning strategy becomes necessary and with that likely.

- filter math500:
	- for any behaviour type, a judge model should decide whether a certain behaviour can be admitted in the solution.
 	- output should be a dict { key : value} where key is the category of the behaviour taxonomy and value \in {0, 1} is whether or not this question allows
 		for natural augmentation.

	
`slack notes`
I'm still struggling to connect the initial task to measure exploration ability
cleanly to the current set of delta log-prob experiments. Here is my understanding:
In an ideal scenario we would like to sum up the probability mass of all traces containing
a certain behaviour like "Backtracking" and look at the sum over the course of training.
What we can reasonably do is to make an approximation of that sum by sampling from this
set of answers we deem would fall within this bucket of traces which show certain behaviour.
The task is now to make as good an approximation as possible.
By choosing to augment ground truth traces we may collect very unlikely completions.
By relying on the results from evaluating the likelihoods for those traces - even when we
average over a lot of questions - deminishes the strength of any conclusion as we invite confounds for any trend.

-> I believe that this delta analysis only makes sense given that the
ground truth response and the augmented response are about equally likely to be sampled.
The further away the augmented trace is from the gt, the less interpretable any result becomes.
-> Any augmentation must be reasonable within the context given.
-> This also means that we cannot just augment any ground truth response. The question itself must
admit to leave room for the augmentation to appear naturally.
-> If the augmentation is too far away from the ground truth we may just measure general degeneration
of the answer rather than the effect we're interested in measuring.
-> This leaves these ways to progress with this idea:
        1) find a class of problems which allow for diverse solutions and augmentation with all effects
                -> This is hard as we're testing multiple behaviours: Reasoning Strategy, Type, Errors...
        2) restrict the set of augmentations such that an application is always feasible
                -> This is hard to defend from a scientific standpoint. This would mean we only look
                at behaviours which are so common, they appear in almost all context. This goes
                against the initial task of measuring exploration (to unlikely domains) to a certain extend.
        3) condition augmentation on whether the problem admits to augmentation.
                -> The statements we can draw from the final analysis couldn't be generalized to the whole
                validation distribution immediately. Rather, we'd look at a conditional data distribution we're
                making statements about.
- working towards exp07:
	- look at whether the rollouts have finished.
		- no they havent. failed probably.
	- find intersection of correct answers. Use those to construct a dataset of ground truths.
	- next: Make the generation script access the rollouts

- check up on exp05 rollouts
	- seem to have finished, but not for all checkpoints.
	- should i restart?
	- i think for preliminary analysis it will be enough.
	
- checkup on exp04 followup
	- running

- I'm worried that the rollouts I'm getting for the augmented responses are just too low quality
	- I could manually collect 3 interesting highschool level questions and generate ground truths for those per model
	- Then, I could manually step in and create the suffixes necessary to isolate the effects
	- => Should be ~3 per category => ~100 samples in total.


#### 11.11
- further experiments
	- towards exp07:
		- need way to generate & grade outputs
		- grading outputs can be done post-hoc generating rollouts.
		- first, i should generate a bunch of rollouts for a multiple data sources
		- should implement --valdata flag in run.py
			-> if left empty should default to current set of validation sets.
		- use the --valdata flag to run rollouts with base models.
			- try base models:
				- Qwen3-8B
				- gpt-oss-20b
				- Llama3.1-8B
			- try data:
				- deduplicated gsm8k variant ~720 samples
				- math500
		
		
- reduce load on disk by not dumping inputs/o

#### 10.11
- experiments
	- analyze first experiments
		- dryruns 
			- deltas
				- reasoning types
				- reasoning strategies
				- error types
			- fully data based
	
	
#### 08.11
- masterarbeit anmelden/ pdfs verschicken.
	- AI Statement

- trainings überprüfen
	- running

- trace augmentation framework started but ran into prblems:
	- got aime25 correct completions from hugginface dataset.
	- got reasoning strategy and reasoning type taxonomy
	- tried to inject behaviour into taxonomy, but seems to be hard.
	=> I still think this is the correct way, I just need the right prompt/model to inject the behaviour.
	-> look at the error classification paper to see how they do it.
	- they use a different prompting strategy. They use 5-shot evaluation
	to use in-context learning to arrive at the correct solution. I could
	do the same.
	- test out whether this works with aime25, but i suspect it wont
	=> Use something easier like gsm8k some middle school level benchmark.
	=> should probably collect diverse tasks, diverse answers.	
	=> In COntext Learning paired with larger models seems to be the
	way to go. I should definitely look at generating these with
	strong foundation models instead. For now its fine.

- generate delta dataset for reasoning types and reasoning strategies.

#### 07.11
- exp02 continued:
	- have to re-start DAPO, GRPO, as I havent saved the checkpoints before.
	- run Qwen2.5-7B models with KL-cov, clip-cov, Entropy-regularization, GTPO, GRPO-S
	
- trace augmentation framework started but ran into prblems:
	- got aime25 correct completions from hugginface dataset.
	- got reasoning strategy and reasoning type taxonomy
	- tried to inject behaviour into taxonomy, but seems to be hard.
	=> I still think this is the correct way, I just need the right prompt/model to inject the behaviour.
	-> look at the error classification paper to see how they do it.
	

- exp03 failed after first logprob dump.
- exp04 also failed after first logprob dump.
	=> should be fixed.
- re-assign work to finish experiments.
	- trace augmentation
		- i should start with a dataset which has got ground truth
		solutions to the questions.
		- quickly iterate to see whether base models are able to
		generate these augmented answers.
		- not all methods of reasoning are applicable to any problem.
		i should pre-filter a set of problems for each reasoning category
		in which the type is applicable.
			=> This brings added complexity. Couldn't I instead just reduce
			the number of reasoning types for now?
			- Pattern Recognition
			- Deductive Reasoning
			- Inductive Reasoning
			- Hypothesis Generation
			- Validation
			- Backtracking
		- Build taxonomy. Use taxonomy to generate cases.


- start larger trainings for less methods.

#### 06.11

- start jobs to test whether the logic works.
	- exp03:
		- aim: look at differences in logprobabilities for the error dataset /u/rfechner/data/eic_gsm8k_generated/delta.jsonl
			between grpo, grpo-s, clip-cov
	- exp04:
		- aim: look at differences in logprobabilities across rollouts in different checkpoints/rollouts for two methods
			grpo, clip-cov
		- setup:
			- have to rename 'X.jsonl' to 'X_rollouts.jsonl'
			
- should rework run.py:
	- logprobs and validation rollouts only conservatively. All important analysis is done later.
	--> towards separation of training and validation.
		- we need to log validation and train rollouts, but we do not have to set valn=1024 during training runs.
		- we do not need to compute logprobs by default
	- make all eval options conditioned on --eval being set. In case we're not in --eval and any of them are set -> throw.
	- clear flag whether to create validation samples or not. Currently we're validating always when validation_Data_dir is set. however: we may
	want to only compute logprobs. Idea: val_data_dir is fixed and determined by the experiment name + project name. We do not have to change this ever.
	-> add additional flag --no-validation-rollouts which skips the validation rollouts.
	=> have to separate further:
		- no-dump-train-rollouts
		- no-dump-val-rollouts
		- no-calc-val-rollouts
		- no-calc-logprobs <- this also skips the dumping.
	-> need this separation as we have situations in which we'd like to not dump
	=> rather than looking at flags, perhaps its clearer to look at use cases:
		- regular training: ok
			- no logprob calculation
			- low validation samples
			- log val/train rollouts
		- full eval:
			- log validation, log logprobs
		- either log-validation or logprobs eval
			- ability to shut off logprobs/ validation calculation
		=> Ability to hard-shutoff validation calculation or logprob calculation.
				
- towards logprob ideas
	- purely data driven analysis
		- construct dataset on the fly
			- go to the provided checkpoint folder
			- grep the validation outputs
			- concatenate them to a dataset
				-> ~630*n*k; n=8,k=50 \approx 252000 datapoints.
				-> thats a lot, but lets see.
		- check correctness of metadata and structure of jsonl.
	- check correctness of metadata and structure from chat file.
	- batchsize of logprob_dataloader must be divisible by world_size/tp_size i think. I could do some multiple of 8.
	- deltas for error dataset
		- should pre-construct dataset
			-> bring into "suffix" form
			-> i think i could just do that once and run experiments.
			-> lets think about this. We have 900 samples, 100 per category. The idea is to run the ground truth chat, then the error chat and measure
			the difference in logprobs for the responses.
			-> seems straight forward. The suffix script only requires "prompt" -> regular openai api schema, "suffix", "error_type", misc information.
					
- I should start large trainings to later go to and compute logprobs for.
	- think about a subset of models and methods which are interesting.
	- bonus if i finish everything:
		- implement random reward for zero-std groups.
		
#### 05.11
- checkpoint handling:
	- there are different ways to implement the idea:
		- in _validate we delegate to another function which loads checkpoints, then calls _validate again <<<
			-> wouldn't have to touch other trainer files
		- in all trainer files, make a case distinction above the load_checkpoint in fit function.
			-> Would mean we do not touch control flow for other evaluations. Note: Single checkpoint evaluations aren't part of the plan anyways, right?

- training dynamics plots for baselines in slack
	- also post policy gradient loss
	- what takes so long in the kl-cov experiments? -> Plot time measurements per method.
	
- towards the delta log-prob idea
	- impl checks in eval s.t. the checkpoint i choose corresponds to the model.
	- fix interaction: When we re-start training and test_freq != save_freq then we get duplicate logs. 
		-> Is this an issue? => See klcov experiments to judge.
		-> everything in val_jsonl is written with open mode "w", so logs are overwritten. -> fine.
		-> in logs.jsonl however, we have duplicate steps.
		-> I could fix this by instead of appending to "logs.jsonl" i just write to f"logs-{step}.jsonl"
		-> could also just try to de-duplicate
- delta idea:
	- For all of the logprob methods, make an overview of which paths the data takes through the script.
	- from this, distill requirements and changes which apply to all log-prob experiments we want to make.
	- three distinct ideas:
		- grid: load multiple checkpoints, load multiple val_jsonl files. No further data engineering required.
		- suffix: load multiple checkpoints, load fixed dataset. eval logprobs only of suffixes.
			- looks very precisely at effects and isolates to the maximum
			- careful engineering required to isolate suffixes.
		- delta: load multiple checkpoints, load fixed responses, one vanilla, the other augmented with behaviour. Calculate the logprobs for both. Compute delta.
			- We're evaluating loads of data, most AI generated anyways. Hopefully "effects will come out in the wash"
	- all have in common: we have to load a lot of checkpoints and possibly re-start the process.
		- feasibility of loading checkpoints -> something like a previously_loaded_checkpoint in the experiment folder to keep track?
		- always load checkpoints from low to high.
		- after checkpoint has finished, update prev_loaded_checkpoint.txt
		- --grid flag -> provide directory.
			- pushes grid env variable to script. script loads prev checkpoint or first
			- then runs regular validation. -> still have to provide logprob path, validation rollouts path.
			- should write down all possible use-cases for running validation. If i'm not careful I'll mix this up.
			- i think it's best to have one flag activated per experiment I'm running over all checkpoints.
				- grid: loads validation rollouts into dynamic dataset for each eval run in provided folder and evaluates flat logprobs
				- suffix: loads specific dataset and evaluates flat logprobs
				- delta: loads specific dataset and evaluaes flat logprobs
				- rollouts: generate rollouts for static set of validation questions
			- I need to condition the creation of the logprob dataset on the type of logprob evaluation.
				-> suffix needs different format than others.
				
`exp02`
- aim: replicate DAPO, GSPO to be sure my setup is good.
- setup:
	- look at DAPO paper. I suspect they train for thousands of steps
	with way larger batchsize.
	
- verdict: DAPO's adaptive sampling makes together with the dapo17k training dataset makes 
	it such that we're taking very long to generate prompts with non-zero std. Currently the
	qwen2.5-7b models are taking ~10 batches of queries to get 50 non-zero std rewards. This must be faulty somehow.
	--> however: when trianing with math, we're getting ~200 non-zero queries per batch.
	--> should i lower the training batchsize for dapo?
		
`exp03`
- aim: further log-prob idea. Generate and collect eval dataset for reasoning indicators
- setup:
	- Generate data for reasoning Indicators.
		-> Could tell a model in system prompt it only solves problems
		using reasoning type x, then generate & grade some rollouts.
		-> Does that produce good results? Could also just handcraft the
		rollouts. 
		-> Should prioritize quality over quantity here. It's okay if
		i just have 3-5 examples per class.

`exp04`
- aim: thaddäus idea on delta between log-prob of correct versus right answer
- setup:
	- Dataset prep -> bring eic dataset into correct format.

`exp05`
- aim: data-centric approach to measuring exploration
- setup:
	- evaluate trained models rollouts on base model -> generation pipeline.

`min goals`:
- abstract
- implement KL-Cov, Entropy Advantage, Entropy Regularization methods
- judge test
	- parse rate 1.0 but accuracy 0.0 ? <<
- generation pipeline 
	-> use main_ppo with validate_before_train and early_exit flag.

`A goals`:
- exp02
- exp03

`A++ goals`
- exp04
- exp05

#### 04.11
- any resulters? óuò
- finalize generation pipeline
- exp03?
- get some graphs, write down experiments i want to make to talk about
	in the meeting.

#### 03.11
- results:
	- dryruns
		- klcov, dapo finished?
		- problems with dapo:
			- We're trying to reach batchsize 512 but we only get ~2 std!=0 groups per trainloader batch.
			- Could try to increase rollout_n to 16 or decrease batchsize to ~50.
			- This shouldn't really happen though. Are other ppl experiencing equal problems?
				=> This is only an issue for weak models like llama3.2-small. For Qwen2.5-7B this isn't an issue.
		- lets plot the dryruns.
			- loss, entropy, accuracy means.
			- identify further possible bugs.
			- policy gradient loss is still weird -> test for larger models.
			- no entropy measures for dapo, kl-cov, clip-cov.
				- likely because I'm popping entropy from batch somewhere in dapo trainer.
				- in dryrun we dont have entropy for dapo, because that run is degenerate.
					for kl-cov/clip-cov the entropy_ray_trainer never adds the entropy to the metrics.
		- klcov/clipcov trained for only a few steps? Why is that?
			- 1) training stopped somehow after 4 hrs
			- 2) throughput is very low ~300 tokens. (?)
			--> is this because I'm training with adaptive sampling?
			--> no. Somehow I'm not seeing any error message appear in the .err file.
			
	- dapo long run
		- dapo finished? -> does it make sense to re-start?
		- currently at step 380, although inflated because of the adaptive-sampling.
		- Should just be able to compare as is.
		- flat compare both runs.
		=> dapo seems to be matching the results by the paper -> It trains way quicker than GRPO.
		-> still work to be done in plotting, but i can do that later. For now at least i can be surer that DAPO does it's job.
				
	- judge runs
		- have to evaluate
		- ~60% accuracy isn't great. How do large models fair?
		- started new rollouts 
		=> Current transformer version errors for Qwen-Next models + OOM errors. Also, for some models we need CUDA compute capability >= 8.9, we have 8.
		=> Should try to look into open router/gemini2.5-pro.
		- take a few samples from the dataset and paste into Gemini/CHatGPT. <<<
		- openrouter api tests.
		-> tested with ~40 completions and got ~65% accuracy for 235B Model.
		-> Should probably still go for the paid models. Also: Look at different prompts.
			-> Let LLMs generate some candidates and restart job.
			
	- eval runs
		- which of both eval runs makes more sense? -> Should probably use another project directory as evaluation output, right?
		- seems to be weird slurm/hydra error. Should try to run from manual file.
		- running manually on one node seems to work until we run into the problem that we've sharded for 8 GPUs in training
		and try to re-shard for 4 GPUs. All tests for this should be done with slurm jobs and two nodes.
		- I've updated the scripts and ran training again. I don't understand why regular training passes config verification, but eval doesn't
		- It was a malformed shell script. I had trailing whitespaces.
		- `running`
		
#### 02.11
- test eval runs.
- check dryrun
	- dapo, klcov+clipcov finished?
- check long run -> should prolly restart but we might as well already see differences in GRPO/DAPO.
- if eval works correctly, i should go to exp03/exp04.

#### 01.11
- --eval mode
	- how to continue from checkpoint?
		- config.trainer.resume_mode == "resume_path"
		- should point to a "global_step_x" folder, and should be absolute path.
		- dataloader is also loaded. Should check whether this is saved for every step.
	
	- automatically set validate before train=TRUE. also set early exit flag.
		- plan:
			- make --eval flag.
			- if --eval is set, we check whether a checkpoint is provided
			- we automatically set the required variables and start training.
			- we early exit after the validation has finished to not overwrite.
			- test.
							
- check experiments 
	- klcov and clipcov
		- import error in config. Expect to fail again soon..
		-> could make sense to iterate faster by making manual version of this.
		
	- dapo failed?
		- error in hydra config, had to prepend "+" to rewardmanager kwargs in config.
		- num gen batches -> should try to understrand better what this means.
			Currently getting error that max_gen_batches <= gen_batches -> data too difficult.
			Why aren't we just falling back to regular GRPO in case this fails?
- judge job restarted.

#### 31.10
- generation script
	- implement --eval flag in run.py.

#### 30.10
- recheck DAPO:
	- seq-mean-token-mean or just token mean loss aggregation?
		- no its token-mean -> correct in my code
	- algorithm.filter_groups.enable has to be set?
		- yes. for some fucking reasong it's not set by default in the dapo_trainer.yaml file.
- Research: Which methods add entropy to the advantage?
	- GRPO-S/GTPO
	- could i implement this myself? Seems like i only need access to
	rewards and entropy on token level.
	- inject this at the reward compuation step?
	

- GRPO-S/GTPO
	- test run
	- gradient norm still nan?
	- no, should work now
	
- KL-Cov/Clip-cov
	- KL-cov and clip-cov have their own trainer file. Should probably deletate to that instead, as they're also using dapo's adaptive sampling etc.
	- test run
	
- GRPO w/ entropy regularization
	- test run
	
- generation script

#### 29.10
- written abstract
- error categorization -> failed again, because llama instruct tokenizers don't seem to have pad token?
- Look at KL-Cov.
	- /u/rfechner/verl/recipe/entropy/7b_kl_cov.sh
#### 28.10

- logs should append to jsonl not overwrite.

#### 27.10
- download datasets to validate text classification pipeline
	- MWPES-300k
	- other error dataset -> look notion
	- Is there a dataset for reasoning type classification? 	
- make prompt, run some tests for different models which fit into
	4 GPUs memory
	- run qwen3-4b baselines on classification. Should prototype quickly
	and just run some models on single gpu. 900 samples is quick
- make open router account and run tests for
- read prasannas paper

#### 26.10
- check on experiments, possible restart
	- LLama models are way faster to train ~ twice the speed.
	Perhaps this is because of Qwen's models is thinking longer and
	rollouts are longer?
- get error mode taxonomy and reasoning type taxonomy going.
- get few good examples from rollouts to test these on
	- why not just take a few of the training rollouts from the qwen and
	llama models?
- pump those into an api or perhaps load a quantized reasoning model from the
	hub.
- should probably also think about how to make sure this approach is
	done in a scientific way: I should create or download a ground truth
	dataset and test my prompts on these. 

#### 25.10

- check on experiments, possibly restart
	- tp=2 does lead to OOM. -> why?
	- validation n does scale gpu memory.
		-> should reduce valn to ~16. Currently it's only important
		to collect good signal on validation, not to calculate pass@k.
	- restart with tp=2, valn=8
	
- reading
	-> form error taxonomy with definitions and examples
	-> form reasoning indicator taxonomy (?)
		- possibly even get more resources
	-> open router free server test
	-> get some plots for single model over training
	
#### 24.10

- carry from yesterday:
	- interpretability of logprob results: "16" is encoded
	into two tokens with Qwen3, but 1 token with llama3. ->
	can we normalize over token length?
	- geometric mean over probabilities
	-> is there a log-prob variant of this?
	-> perplexity

- checklist for regular training runs:
	- logs working 
	- logprob evaluation working
	- rollouts working
	- checkpoints there
	- loss decreasing
	- dapo sampling working
		- manually check whether this is conditional on
		some set parameter.
	- put dapo reward manager in grpo, drgrpo.

- see whether we can run small models with tpsize=2 or even tpsize=1
	-> lets run with tpsize=2
- go over parameterizations, map out first experiment, create sh script
	-> recheck with notion.
- run experiments
- invite prasanna and thaddäus togithub

- take a look at openrouter api
	- can't find student discount?
	
- read relevant papers
	- failure modes
	

#### 23.10

- check all baselines have completed without errors 
	+ we can see learning.
	- not all methods seem to have the val-core metrics. -> why?
	- in grpo and drgrpo we cannot find val-core metrics, only best/worst @k
	- difference lies with the dapo reward manager, which is active
	for gspo and dapo. It is better integrated with current validation
	formatting. I could just switch to the dapo reward manager on grpo/drgrpo
	and set overlongpenalty=False.
	- this doesn't explain however why I'm not seeing reward/mean@k in grpo.
	=> for the regular reward manager we're computing 
	`reward` ('val-core/math-ai/math500/reward/mean@8') metric,
	for the dapo manager we're computing 
	`acc`(val-core/math-ai/math500/acc/mean@8) metric.

=> pgloss seems weird. We're training though.
=> entropy for drgrpo is massive. Is somehow an normalization error?

=> look at multiple validation set run <<<
	-> restart
	
- finalize logprob calculation:
	Should also include the prompt suffix "Let's think step by step and..."
	-> Separate (a) user message, (b) part of assistant
	i don't care, (b) part of assistant i do care in jsonl and in code.
	-> probably first append (c) to the chat, then tokenize (c) alone,
	then tokenize everything, then compute the correct response mask.

	
- generate script
- find resources on error taxonomy, reasoning indicators.

#### 22.10

- what does the dapo-reward manager do?
	- shouldn't do much other than regular reward manager. Just applies
	some overlong penalty on top of regular reward function result.
	
- get some useful logprob completions.
	- search through rollouts 
		-> Get qwen response, llama response.
		
- download scripts for uncontaminated math benchmarks, integrate into
	naive reward manager, also append as validation files:
	- aime25
	- HMMT Feb2025
	- BRUMO 2025
	- SMT 2025
	- CMIMC 2025

	- currently issue with having multiple validation files:
		- different sizes in rewards versus scores?
		- only one reward function for all sources? -> math_reward
		- dapo compute score function returns dict, whereas math_reward
			returns float.
		- seems to work now, but: validation sets other than math500
			seem to have zero mean@1. Check out whether math500 and
			aime25 format is the same.
			=> no reason this shouldnt work.
		- what does the dapo reward manager do other than the default
			manager? I think i should just start a dapo job
		
			
- reward manager seems broken. I'm getting zero reward on the validation set
	- but only for validation? Seems the trainings do compute the reward correctly.
	- dig through training set rollouts to see whether we have positive rewards
		- can confirm: rewards in rollouts for grpo seem valid.		
	- already saw diggin through the outputs that we've got issues with
		parsing the reward correctly: \left( x \right) != (x) when parsed.
	- `validation dataframe has ground truths as full answer, while
		training dataframe has gts as solution. Is there a mixup?`
	- math500 was downloaded incorrectly and in a previous version of verl
	the validation reward manager seems to have included additional checks to
	alleviate this.
		=> remove and download all relevant datasets again. Make sure that
		 "reward_model" -> "ground_truth" is just the answer in latex.
		 
- dapo has problems with rollout compilation? 
	-> systematically search for cause. Seems to be in the vllm setup.
	-> Is this some weird torch dynamo setting in the env variables? 
	-> seems to be the VLLM_USE_V1 flag which isnt compatible.

- start treaining runs for all methods to cancel later. Just want to see whether
	they are training and loss is going down. Set epochs to 1 for this experiment.
	
#### 21.10

- check gspo, dapo
- write down meeting log in notion, ping on slack
- write todos in notion
- write in-progresses in notion
- write simple chats -> grep some completions of a qwen3 model,
	switch around some numbers.
- start trainings with single model qwen3-medium. Collect 512 rollouts per
	checkpoint, run over simple chats

- rewire dapo_entrypoint.sh to point to recipes/dapo/ray_dapo_trainer.py
- create gspo_entrypoint.sh
- make sure we're logging training generations as well.
	- config.trainer.log_val_generations ?
	=> config.trainer.rollout_data_dir != None

- check later whether jobs are still running.
	- failed due to lexer error -> passing '["console", "file"]' seems to
	be an issue?
	- failing to unset ROCR_VISIBLE_DEVICES -> why am i running into this
	sshit again?
	- likely an issue with not the environment variables in the ray runs.
	I suppose ray initializes a clean environment and needs to be passed the
	"ROCR_VISIBLE_DEVICES" manually inside the environ_vars in the
	main_dapo:run_ppo function.
- GSPO Paper (low-.prio)
- how to judge rollouts? -> read literature
- download and test dapo17k: open-r1/DAPO-Math-17k-Processed

#### 20.10

- check if adaptive sampling is executed in dapo currently
	- do we have to change to dapo_trainer.py?
	- likely yes. Do we have dynamic sampling in the regular GRPO code?
	
- we have to delegate to dapo_trainer.py. Hopefully not a big deal, as only
	the fit function is overwritten in the dapo trainer.
	
- gspo?
	- gspo setup seems to be expressable as regular main_ppo.py setup. 

- should see to that also graded training rollouts are saved.
	-> set trainer.rollout_data_dir
#### 18.10

- read and discuss alberta plan, send mail to peer
	- don't really have any interesting papers to share yet.
		- could still ask if peer has any leads?
		- unsure whether the alberta plan for reinforcement learning agents
			is a suitable higher level plan to embed into, as it's
			design philosophy is very idealistic, theoretic and purist.
		- would like to develop neurosymbolic systems which can be applied
			in many different settings. -> Method first.
	  		--> Wouldn't it also be nice to be able to work towards
	  		a problems solution, i.e. "solve ARC-AGI 2" ?

- have the most recent runs logged metrics as intended? -> should be 
	under logs.jsonl
	- experiment name fucked. 
	- should export experiment name in every run script. Does this change
		any other behaviour? 
	
- check whether --cont works, whether logging works
	- wrong experiment name passed to ray trainer.
	- cont seems to work. But check again that second moxin run started
		from step ~40.


TODO:		
- do i load reference model/critic with current grpo setup?
- should i try to switch to dapo_trainer.py instead? Could i just port the dapo_trainer.py
relevant sampling logic to the regular ppo_trainer.py?
	-> chatgpt compare fit() functions of both trainers.
- what about gspo and the other baselines? should i also delegate to another
	train srcipt for those? The issue is management.
#### 17.10

- ran all experiments:
	- grpo - qwen2.5-1.5b - verl
		- finished 14.30h
	- grpo - qwen2.5-1.5b - flashinfer
		- finished 14h
	- grpo/dapo/drgrpo - qwen3-0.6b - verl
		- error -> download model checkpoints first.
	- grpo - llama3 - verl
		- finished 10h
	- grpo - moxin - verl
		- step36, then terminated.
	- grpo - r1-qwen - verl
		- step 51, terminated

	=> Most importantly is looking at whether all methods finish and logging of rollouts, logging of logprobs is working.
	Next up would be whether i can continually train, as this is required to finish my pipeline.
	=> Also: Look at the performance/throughput and timing and whether we can see a difference between flashinfer/verl environements.
	=> Conditional setting of parameters? -> i imagine tp-size cannot always be set to 4? 

- dapo, drgrpo failed. why?
	- dapo: JSON decoder error in load_model_checkpoint_shards or sth.
		-> downloading the model checkpoints has to be done before starting batch script.
- regular groṕo seems to have finished. Look at whether all files are there.

- download all models from hub
	- qwen3-4b is an instruction finetuned model?
	- Qwen3 doesn't show incomplete model files. -> Should try to load in interactive session, then as part of manual run.
- look at disk space.  
	=> put model checkpoints into /ptmp/rfechner

- restart experiment with moxin to try and see whether --cont works as expected.
	- /ptmp/rfechner/out/default/moxin_7b_instruct/val_jsonl -> last validation was number 40.

- should try flashinfer with higher rollout numbers -> 64 rollouts, 1 epoch.
	- two llama3-8b models.

	- look at throughput metrics of the regular and flashinfer runs.
		- i'm not logging metrics and validation accuracies?
		- should add file logger.
	- see whether the export of console+file logger is used correctly.
#### 16.10

- check: Are we masking the computation of logprobs correctly? -> Current answer to query "Whats 2+2" with answer "4" gives two logprobs. 
	There is probably some index shifting we need to apply.
	-> save out batch and response to decode
	-> response mask should be working as expected. Perhaps i could just append the input batch passed to the
		actor to the return batch for debugging?
- torch.no_grad() on the compute_log_prob stuff.
	-> this already seems to happen. Still to be safe i should wrap into torch.no_grad()

- test run -> can we also save rollouts? `<<<`
- what tf is taking so long?
	-> look at timing stats.
- still need to append the "Lets think step by step" to the prompt on each query. I have to make really sure that I'm measuring
	abilities WITHIN the right data distribution, i.e. it's very unlikely that llms will generate "4" immediately after being
	instruction finetuned. Hence, signal of just measuring "4" instead of maybe appending something like a Generation prompt
	"To solve this... Finally, I have come to the conclusion that the result is $\boxed{4}$." Would be more likely to measure correct
	behaviour within the "reasoning" data distribution.

	- nice side idea: Could we show that language models are only stable "calculators" within the right data distribution?
- Moxin/Llama stuff.

- ran all experiments:
	- grpo - qwen2.5-1.5b - verl
	- grpo - qwen2.5-1.5b - flashinfer
	- grpo/dapo/drgrpo - qwen3-0.6b - verl
	- grpo - llama3 - verl
	- grpo - moxin - verl
	- grpo - r1-qwen - verl

	=> Most importantly is looking at whether all methods finish and logging of rollouts, logging of logprobs is working.
	Next up would be whether i can continually train, as this is required to finish my pipeline.
	=> Also: Look at the performance/throughput and timing and whether we can see a difference between flashinfer/verl environements.
	=> Conditional setting of parameters? -> i imagine tp-size cannot always be set to 4? 

#### 15.10

- Go to RLDataset and implement log_prob flag -> should apply chat template in a different way.
	- take inputs, outputs and merge them into format [{"role" : "user", "content" : "What's 2+2?"}, {"role" : "assistant", "content" : "4!"}]
	- check whether this works in debug mode first.
	-> create logprob dataset and take first element.

	debug:
	- failed assertion: _split_args_kwargs_data_proto: passed value isn't DataProto.
		- some keyworkd argument isn't a DataProto. Could this somehow be missed inside the from_single_dict method?

	- update:
		- can't pass regular training batch into compute_log_probs, as the function is designed to take as input
		input + response already. Have to append the response tensor dict to my batch, conforming with regular
		structure. -> Look at how the compute_log_prob batch in the "compute_old_log_prob" section is constructed.

		- batch is unified with gen_batch_output. -> THis means that it has the response as key.
		- fastest would be to just mimic the batch before its passed into compute_log_prob. For this, I'd need to run training without val,
		dump the gen_batch_output and regular batch.
		- should just separate inputs and outputs into prompt and responses. -> Should be done within the dataloading routine.
		- in the data batch which is sent to the compute_log_prob function:
			- attention_mask is 1 at prompt + response. Is the response-mask added somewhere else?
			--> Has to be calculated extra. -> see compute_response_mask.
	DONE
	

- Moxin/LLama configs -> chat template is faulty.
- Why is the code so slow?
	- move back to brendels version of verl and try to run there.
	- re-install verl clean and try to run on both versions of verl.
	
#### 14.10

- validation data isnt data proto?
- moxin and llama: tokenizer chat template isn't passed.
	-> https://huggingface.co/docs/transformers/main/en/chat_templating
	-> how is this handeled with regular qwen models?
	
#### 13.10

- restart jobs:
	-> separate the two flags for rollouts and logprobs s.t. i can double debugging speed.
	-> moxin/llama models download scripts

#### 10.10

- currently having problems with calculating the logprobs for the perplexity idea
	-> Don't really know the canonical way to just push something through the worker group.
	-> Have enabled auto_padding=True -> lets see whether that helps.
	-> Logging the validation completions comes after the log_prob_dump so lets see.
- did the llama and moxin llms start training? Do i have to make some model-specific changes to the configuration?

#### 08.10

- finish training pipeline:
	- add baseline trainings
		- entropy regularization
			- Clip-Cov
		- Dr.GRPO / GRPO++
		
	- verify smooth runs
	- advantage logging
		- can we collect trajectories on the fly?
		- which hyperparameter controls the n in validation responses?
		- in which format are the answers logged?

- begin working on evaluation pipeline
	- think about for which of these evaluations we can share evaluation compute
		1) 
			- loading checkpoints to assign log-likelihoods

		2) 
			- rollouts for regular validation questions
			- generation entropy
			- expansion/shrinkage
			- embeddings		
#### 06.10

- have to clean house for experiments to start. I'm thinking of just one python entry file, but many sh files to launch the batch script.
	- options: GRPO, DAPO, Dr.GRPO
	- find common settings

- i have to work out the overarching experimentation pipeline. Which dimensions do i wanna test? models, algorithms, data?
#### 03.10

- get dapo running.
	- performance tuning in manual_train.py
	
#### 01.10 - 03.10

- performace tuning. https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
	-also: can i somewhere see the plot for gpu memory consumption over time? would be nice to see if we can increase throughput. 
- was brauche ich für einen ersten mock versuch?
	- welche statistiken muss ich während des Trainings akkumulieren um eine klare Trennung
	von Versuchslauf und evaluation zu erreichen?
		- entropie logging
	- verl hat eine option für "advantage_logging" oä. Diese option sollte ich mir anschauen bevor ich
	den Plan fasse die completions von checkpoints aus vervollständigen zu lassen. Wahrscheindlich kann diese
	Option nur bedingt die verschiedenen Messwerte abdecken, welche ich für die Analyse geplant habe, dennoch
	ist es sicherlich einen Blick wert. Beispielsweise könnte ich somit sofort den pass@k wert berechnen lassen und
	die completions für das test-set generieren, graden und speichern lassen.

#### 30.09

- omni-math paper
- plan research for this week. Work towards prasannas idea. Get first prototype going.	

#### 24.09

- random reward for zero std groups
- adaptive advantage scaling.	
#### 23.09

- questions about the entropy masking:
	- can we see the entropy go up in the neg-mask case?
	- can we see entropy go donw in pos-mask case?
	--> There seems to be a mixup in mask-neg and mask-pos. Mask-pos makes
	the entropy go *up*, whilst mask-neg makes entropy go *down*.
	--> for now just switch sign of H_t to keep naming scheme and make comment. 

- implementations of further ideas.
	- entropy change:
		- look at H_ts over time, look at advantages over time.
		- i think there is still room for simple experiments. I've shown that
		my mechanism works and i can increase or decrease the outout distr. entropy
		by masking positions based on first order approximations to 
		changes in entropy. Whats missing is a tradeoff mechanism which switches or
		linearly interpolates between those two modes. 

		The theoretical setup is: I have B sequences of L^(i) floats. I want to dampen
		or scale the entries in these arrays, such that a target entropy-change level
		is reached. Plan: For each epoch, set a target in the range [-1, 1]. This could
		be calculated based on a scheduled plan, like 9 episodes results in the targets
		{-1, -0.75, -0.5, ..., 0.5, 0.75, 1}. Based on these target values, we'd determine
		a positive and negative change by summing over the predicted induces entropy changes
		H_tpos and H_tneg. Then, we'd equate alpha * H_tpos + beta * H_tneg = target.
		if we also require alpha + beta = 1, then we have two knowns and two equations,
		which we can solve.

		TODOs:
		1) write infrastructure to make adaptive advantage scaling possible.
			remove early stopping for these experiments and set epochs to 10.
			do not need a model save just yet.
		2) in code, solve the equation based on H_tpos, H_tneg, t.
		3) start tests with different target schedules.
			- [-1, -1, -1, ..., 0, 0, 0]
			- [-1, -0.9, ..., 0, 0, 0]
			- [-1, 0.9, ..., 0.1, ..., 1, 1, 1]
			etc.
		
#### 17.09

started 1.5B experiments, started run to figure out whether tpsize=8 will allow 7b
training without sequence packing.
#### 15.09

- wieder in die experimente finden.
	- warum sind die letzten experimente fehlgeschlagen?

- read: 
	- The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
	- Outcome-based Exploration for LLM Reasoning (?)
	- EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction for Advantage Diversity
	
- kann ich entropie in einem versuch vorhersagen um meine berechnungen zu verifizieren?
	- quick prototyping mit ChatGPT.


abfolge:
	- interaktive session starten
	- herausfinden warum die experimente fehlgeschlagen sind
	- wechselspiel: experimente starten und paper lesen.
	- ziel heute: experimente zum laufen bekommen und starten, 2 paper lesen.
	
#### 22.08

- read new papers
- brust training

#### 21.08
- check out single batch training entropy deltas alignment with true entropy deltas.
- check out whole training run
- if predictions are okay:
	- think about how to use the deltas.
- if predictions aren't okay:
	- think more about how to predict entropy change.

#### 20.08

- check contract extension
- check out connection between entropy gains and advantages. Why are so many
	entropy changes zero?
- Plot entropy changes over course of training...
	- sign error?
	 
#### 19.08

- weirdness with GPU memory, saving to pt format.
- look at the entropies later
- prepare meeting messages today:
	- Training is normal. Numbers validated by another paper also using verl.
	- Show pass@k for math test set. 

#### 18.08


- why havent the checkpoint rollouts finished?
	- they all fail when processing the last batch. This is like in previous
	runs when I had to split the math500 dataset into two batches to be processed.
	-> `Currently running with tp=8 to see whether that changes things`. 
- entropy dumps:
	- have to print debug information. shapes of tensors etc. Currently limited
	by memory. I think this may be an issue when we switch to 7B models.
	- print batchsizes response_sizes etc. In non-pad-remove mode we probably have
	ALOT of unused space in memory since max_response_length < response_length usually.
	- go on gpu node, see whether training dumps entropy deltas
- make new commit to my branch.

#### 14.08

- check whether it is computationally and technically feasible to collect the
	estimated changes in entropy per update.
	- we collect entropies and log-probs inside the dp actor. In there, we could
	as well re-balance the entries
	- `note: we do not get the log-probs for the non-sampled tokens`
	if we wanted those, we would likely blow up memory. Just get the log-probs
	for the top-k logits instead to approximate entropy change?
	- dp_actor.py -> line 126
	- i should think more about what the ultimate outcome for this analysis
	might be.
		- re-weighting scheme? We would re-weight traces s.t. we can balance
		the effect on entropy.
		- penalize/push up individual token-level advantages to compensate for
		entropy loss
 
- check the rollouts and plot the pass@k's for the math validation set.
	- possibly re-start the rollouts which didn't finish
	- restart the chunks which didn't make it

- Run GRPO training and look at rollouts for checkpoints
	-> save every 2 steps, run for 20 steps in total. This should be two epochs
	and already arrive at a pretty good model. It's not necessary to get the best 
	model but i think we should be most interested in generation entropy
	and these diversity analyses in the first few steps of training.

- should try to debug in interactive mode to get the shapes of the tensors. Alternatively
log everything...

- mail for contract extension?
	
#### 12.08

- write prasanna when to meet today
- make run_rollouts script to take in specific chunk numbers to retry
- load prasannas rollouts
	- /lustre/scratch/pmayilvahanan/post_training/verl_checkpoints/rl_ood/evaluations/

prepare meeting:
	- deltas histogram visualization
	- visualization of 1.5b models
	- visualization of 7b models
	- visualization of embeddings, sentence embeddings
	- need to write down what setups we want for analysis exactly.
	
#### 11.08

- restart rollouts for 1.5b with --model option.
- take a look at todos for tomorrow.

- sentence based clustering (low prio)
	- mathematical reasoning isn't linear - two responses may arrive
	at the same location by different strategies. Further, two very similar looking
	answers may be completely different. Can we come up with a better measure
	for strategic diversity for the answers of LLMs? 
	- use Optimal Transport library to compute cost of "moving" one
	sentence to another. This way, we might be able to better cluster
	responses.

#### 10.08

- make it possible to pass tensor_parallel_size through cli for
	math500 datasets...

- rollouts:
	7b:
		math500
	1.5b:
		minerva
		olympiad
		aime25
	

#### 09.08

- start rollouts for best_checkpoints of all models in "fixed" batch
- start rollouts for entropy regularized "new" batch

#### 08.08

- check that all trainings have terminated by early stopping, not by OOM
- fix best checkpoint saving
	- logic should be rather simple: Save the previous local dir, Set the current local dir of
	the checkpoint manager and call the save_checkpoint function. Afterwards, restore
	the previous local dir
	- I don't understand why the "best_checkpoint" directory is empty! Also on some runs
	it seems like the algo has determined a best checkpoint, but saved in the
	global_step_X directory rather than the "best_checkpoint" dir.
	- ray workergroup doesn't have the attribute "checkpoint_manager"
	very weird.
- start rollouts for Qwen 7B GRPO and Qwen 1.5 GRPO.
	 
#### 07.08

- check out new training runs' loss and evals. Early stpping has terminated training
	after 10 steps which means that either early stopping is bugged (perhaps
	we calc metrics for the same val output for mutliple steps, as we're just
	validating every few epochs..)

	- 7B
		- GRPO-0
		- GRPO 0.01
	- 1.5B
		- GRPO-0
		- GRPO 0.01

`training runs`
- make it s.t. whole global_step archive is deleted, not just pt checkpoints.
- any chance we may get a good early stopping callback?
- trainings do not finish reliably. Have to re-start GRPO-0.0 and GRPO-0.01.

	
- run rollouts
	- delete unnecessary files like corrupted trainings etc.
	- make sure to unify naming schemes and locations on disk
		- place "out/rollouts/new_Qwen" to "out/rollouts/Qwen/Qwen... new..."
			according to the current naming scheme.		
	- okay so it seems like the training doesn't finish the whole 15 epochs,
	but i may as well just look at whether the trianings have converged enough.
	I should look at the GRPO, GRPO-0.1 and GRPO-0.01 for convergence.
	- after making sure that the trainings have converged, I should look
	at running the rollouts. I want rollouts on math500 for now, when they finish
	I could look at the others. I should also make sure to add the identifier "new"
	to distinguish the new rollouts from the old ones.
	- I should also run the rollouts for the 1.5B variants
- continue with embedding stuff <<<
- für neuwied packen

#### 06.08

- running the trainings with group size set to 8 again under the id prefix "new"

- check whether the GRPO setup is as vanilla as possible.
	- compare to GRPO paper
		- GRPO paper has KL coef 0.04 and group size 64, training bs 1024.
	- compare to verl default grpo setup
		- kl_loss_coef 0.001, group size 5, 
	- compare to prasannas setups
		- kl_loss 0, group size 8, weight decay 0.1
	- compare to base setup of PPO
		- rollout temperature 1.0
	- i think i should ask a chatbot to retrieve relevant comparisons

- embedding fun
	- very interesting results. GRPO (bugged training run) responses do show
	interesting clustering beaviour: They are almost ontop of each other in some cases.
	I suspect its because the answers are all the same.
	
#### 04.08

- create olympiadbench entropy0.01 df and extend experiments
- re-check that the training setup is vanilla
- cleanup and further the analysis notebook
	- plot histogram of deltas
	- which sorts of questions degnerate? -> (type/level)
	- are rollouts for math500 already finished?
		- if so, look at whether the same questions are un-learned.
	

#### 03.08

- meeting notes:
	- LLMs are bad in recognizing failures in their own generation
	- ideal: Supervisor model gets extra context -> correct solution or supervise finetune on correct classifications
	- using larger model is decent solution (o4-mini -> high)
	
- todos
	- check on experiments and start the second round of rollouts
		- experiments keep getting cut off. Find out why.
		- likely a memory issue. Keep running out of memory on the A100s.
		- reran olympiadbench-entropy0.01

		- run math500 for GRPO/ e-grpo/ seeds {0, 1}
			- ran everything apart from GRPO seed 1, which failed after 5 steps. 
				-> reran training
	- write an abstract for a paper
	- read through the abstracts of ameyas papers <<<
	- look for good datasets to RLVR a model for failure mode categorization.
	

#### 02.08

- abstract ideas:
	- what we'd actually like during end of training is more exploration rather than exploitation. However, most questions
	tend to be solved better and better during training. Some are solved not at all and for some questions, the model
	seems to forget how to solve them. I'm interested in the progression of the solve-rate per question during training
	and in the impact of the policy updates. 
	- percentage of positive-rewarded answers per question over course of training
		-> could we find an imbalance there?
		-> analysis of positively rewarded trajectory impact output distribution versus negatively rewards
			-> I think there is a paper which already looks at that.
		-> Could we somehow re-weight the gradients of positively rewarded answers based on their similarity?
			-> this is somewhat done by DRA-GRPO
	- model failure modes: Is there a pattern in failures? Does the model unlearn certain solutions because its training
	data doesn't contain enough basic examples? Is the fogetting systematic? Are the same questions un-learned?
		-> For this I should probably run multiple trainings and compare.
		-> is there a correlation between performance-on and contained-in-pretrain-data of categories of math questions?
			-> positive correlation would mean you have to have a broad and diverse posttraining corpus in
			order to maintain performance.
	- Look at impact on gradients if ngrams overlap strongly
		-> I'd guess this creates these `islands` of low-uncertainty where the model generates the next token
		with high probability until arriving at a forking token.

- start training jobs
	- GRPO 2, 3
	- GRPO entropy 2, 3

- start rollout jobs
	- base -> math5000 questions



#### 01.08

- extend MPCDF cluster access -> write anastasiya

- think about the gradients of the questions during the course of training and how
	'easy' questions influence ngram-probability (Lets evaluate this in python...) on different questions.

	- analysis requirements:
		- Compare questions which are are high passrate in base, but not in GRPO (entropy0.0)
		- Compare questions which are are high passrate in base, but not in entropy-regularized GRPO (entropy0.01)
		- Compare the before-and-after splits of questions and how they "migrated" from being solvable to unsolvable
		or vice versa. I imagine it would be nice to compute the differences in passrates between the two setups and
		yield the top 10 changes in passrate delta and the bottom 10 changes in passrate delta. These questions are
		of particular interest.
		- Identify questions which are hard to solve in both setups (GRPO and entropy-reg-GRPO) but easy to solve in the base model
	
	- There are some questions, which the base model can almost never get correct and there are questions,
	which the base-model already solves perfectly. Both of these cases are proposed to be released from gradient
	contribution, as their advantage will be zero anyways. I'm interested in which questions contribute `most` to
	the policy change - pushing the model into regions which use a particular strategy to solve problems.
	
	- I should look at questions which are solved very rarely and construct hypothetical groups and gradients thereof
		- define a metric which tells me the semantic similarity of the responses.
	- Sample some completions of these sorts of questions with high and low temperature.
	
#### 31.07

- rollouts analysis
	- check whether all questions are present. rerun for incomplete questions.	
#### 30.07

- rollouts for entropy models
	- collect rollouts for entropy0.01. Entropy regularization of 1.0 seems to
	have degenerated.

	- basemodel `done`
	- GRPO
		- math500, minervamath, olympiadmath, aime25 `submitted`
	- GRPO-entropy0.01
		- math500, minervamath, olympiadmath, aime25 `submitted`
- read papers:
	- Error Classification of Large Language Models on Math Word Problems:
	A Dynamically Adaptive Framework
	- Can LLMs understand Math? Exploring the Pitfalls in Mathematical Reasoning

#### 29.07

- I'm probably overtraining these models
	- early stopping?
	- not implemented in verl...
	
#### 28.07

- entropy still collapses even with entropy regularization coef=0.01. Rerun with coef=1
	- reran with entropy=1.0
	- `have to run with entropy=10.0`
- run rollouts with GRPO/entropy=0.01
	- base
		- math500, aime25, minervamath 
		`-> already in verl/workspace/judge/graded_results`
	- GRPO
		- math500, aime25, minervamath
		-> done

	- GRPO-entropy-0.01
		- math500, aime25, minervamath
		-> done
	==> thees models are trained on 8 gpus, hence their pytorch model checkpoints
	are split. I have to either collect rollouts on 8 gpus or somehow merge the 
	checkpoints and re-load on 4 GPUs.
	
- plot histogram

#### 27.07

- cleanup old files and model instances.
- find out why one job took 120 steps while the other only took 30 steps
	- running with 2 nodes and smaller batchsizes seems to let us train
	way faster. 4 nodes with higher batchsize also trains, but way slower.

- rollouts for math are somehow quite slow on some nodes, on other nodes they finish quick.
	- alot of chunks didn't finish at all
	- I think i should instead just try to make the number of rollout instances
	equal 4 on these nodes and then run in parallel.
	- disk quota was exceeded...
	- re-run rollouts
	- measure time for completing a batch (math500) with tensor_model_parallel_size=4 versus 2.	
				
- rerun training with regular GRPO, entropy regularization=0.01.

#### 25.07

- check train scripts -> grep train2, train4
	- ray address should not start with http -> check train2
	- 2 node setup makes steps! `running`
	
- check gradings and prompt optimization.
	- prompt optimization running
		- baseline seems best now...
		- should definitely rerun with higher number of samples to be sure.
		we're especially interested in mislabeling and wrong tool use.
		- I think we will not get around chooseing a prompt based on
		the accuracy of the resulting classification. It seems just being guided
		by prasing rate is the wrong approach. I should go through the first 20
		samples and choose a ground truth error mode for that, based on that I should
		go and re-run the prompt optimization.
		
	- gradings failed: activate verl before.
		- reran, now `running`.

- run larger gradings and look @ whether the majority vote thingy makes sense.
	- finished. Analyze results <<<
		- alot of mis-categorizations. Perhaps I have to use a stronger model.
		- looked at using Qwen2.5-14B 
		-> seems to also not work.
		
#### 24.07

- 7b train 
	-> see whether the 4 node version actually trains faster.
- llm as a judge
	-> wrote a parser
#### 23.07

- make the drawio
- 7B training
- LLM-as-a-judge doesn't really work well with categorization of failure modes.
	- keyword matching.
	
#### 22.07

- make notion section for meeting. <<<
	- what did i do the last 2 weeks?
	- whats my plan for the next two weeks?
	- what are the roadblocks?
	
- make drawio with the different research directions.

- Read FIRE sampling paper
- is it possible to have "bursts" of temperature in sampling?
- is it possible to make <dream> </dream> context with higher sampling temperature?
- research on perplexity of answers which are semantically identical

#### 21.07/22.07

- should organize the folder structure a little bit better, as the rollouts
	were placed unfortunate. <<<
- rerun experiments
- sort by passrate & look at the questions/answers of the LLM
	- why do LLMs fail? Is there a pattern to the failure? 
	- How may we judge or categorize the reasons for failure automatically?
		- I could push the question, LLM answer and the ground truth answer to
		an LLM and categorize this way.
	
		 
#### 18.07

- bug in main_generation, chunks weren't getting chunked correctly.
	- fixed, started again.

- training of Qwen 7b model. -> 8 gpus. -> 2 nodes
#### 17.07

- chunk enumeration also has to incorporate the SLURM_IDX.

- can we load from a checkpoint and then do the rollouts?
	- write small config for testing and set the model.path to the checkpoint dir
	- seems like verl doesn't support loading a rollout from checkpoint yet.
		- did write a little piece of code in the RefActorWorkerGroup to
		load from checkpoint on init. If this doesn't work, i should really
		consider either:
			- making the worker group an actor and change the config accordingly

#### 16.07

- start training for Qwen1.5B models with different entopy coefficients `DONE`
- how to write a job-array in slurm? How may i give it a model name, datatset name etc?
	- slurm job arrays.
	- use python to start these scripts would be the best i think. I can use a single entry point which does:
		- take in the model
		- take in the dataset
		- automatically chunks the sbatch


#### 15.07

- rollout scripts seem to not be finishing. I need a solution which dumps the batches
	instead of collecting everything and dropping at the end.
	--> write a script which dumps "part" files, then stiches them together afterwards
	- test with low batchsizes
	- rollouts are very slow
	=> increased the batchsize in the main_generate script. performance got a bit better.

- rollouts with vLLM are super slow somehow. I'll be using the regular transformers
	rollout script. -> Now is probably the time to write a little bit of code
	to be able to use configuration files for rollouts.

	=> I'd like to use my configmanager.
	=> Generation main config.

- re-run rollout collection scripts for lower nsamples and nquestions
	- first clean up job
	- see if i can pump up the nsamples
	
- look at whether i can load a trained model and collect rollouts.
	- why doesn't my script save the model binaries?
		-> it's because save_freq=25 but I'm training for less than 25 steps appearently.


- run full rollout scripts for math Qwen7b and Qwen1.5b
	- for Qwen7B 512 rollouts for 128 questions took ~1h
	- for Qwen1.5B 512 rollouts for 128 questions took also ~1h
	- in the meantime, i might want to take a closer look at the questions
	and which are 'hard'. Generally, these questions also have a rating attached to
	them. We may plot pass@1-rate versus given level to see, whether human labeling
	does correlate with the passrate of the model.
	- also might want to take a look at medium-hard questions and see, whether there
	are `diverse` solutions popping up.
	
#### 14.07

- started the collection scripts
	- do the scripts terminate within 24hrs?
	- script to sort questions by their pass@1 rate
		- look at questions which are "easy" and look at questions which are "hard"
		-> Why do the LLMs fail the hard questions?

- different klcoef trainings
	- whats the point? Should we compare the outputs afterwards?
		-> I guess the hypothesis is that training with KL-regularization
		doesn't allow for too strong distribution sharpening - we conserve low-likelihood
		regions of the output distributions.

	- which KL divergence approximation do we use? Does it's gradient actually
	approximate the true KL-div gradient or the reverse-KL-gradient?
		=> use the regular "kl" monte carlo approximation and "low_var_kl",
		which's gradient is equal to the reverse kl-div.

	- training and saving is quite disk-space intensive, i should be careful to
	set the save_freq correctly. -> Should calculate the training steps as
	the number of num_epochs * (num_datapoints/batchsize)
	--> how many epochs does prasanna usually train? -> 22 or 200 (?)
	==> num_datapoints: 22 * (5000 / 512) \approx 220 steps
		-> Could i also just backup on training termination?
	- which is the correct coefficient? kl_coef or kl_loss_coef?
		==> kl_loss_coef seems to be the right choice judging by the comments
		in the default configuration from verl. https://verl.readthedocs.io/en/latest/algo/grpo.html
		(search for "kl_loss_coef")

	 
#### 11.07

- simple training run was successful.
- Next: evaluate pass@k=100 instead, run for longer. 
	- collect the evaluation outputs.
- look at the responses and the format. Is it alright? What about scoring?
- what about multi-gpu workloads?
	- next: try to run on 4 A100s Qwen2.5_7B -> 8 Questions, 16 rollouts

#### 10.07

GOALS:
	- figure out robust logging, checkpointing and such
	- start training with SLURM -> different KL-coefficients
		
- Where can i see the average reward? Where can i see the loss go down?
	-> I think in the rollout script, there isn't any training happening, 
		the output looks way different than on the training scripts.
	
- the main_generate script should be used to collect rollouts.
	-> Try model=Qwen/Qwen... instead of actual path.

#### 09.07

- trying to access other cluster
	- have to put my work on a branch, then publish the branch, git switch to that branch on mpcdf
- should establish routine workflow. Tunneling seems to not be supported with MPCDF. should look into that.
	-> try to get the tunnel thingy going today. Training run on mpcdf seems to work fine.

#### 02.07

- Should rather look at the AIME 25, ... (look at notion) benchmarks instead of gsm8k or lighteval.
- Just get a full node instead of single GPU. This should fix any weird ray problems.
- look at prasannas code instead, just try to run it

#### 01.07

- Looking through the lighteval-math problems, what sort of diversity in the solution-space am i looking for even?
	-> there isn't really a concrete thing that defines many diverse solutions, we can just say "this solutions is different from the other"
	-> 


#### 30.06

- collected answers for a single question from gsm and math codebases.
	- should probably first skim over questions
		-> sample 20ish questions, render to latex and just go over them (01.07)
	- collect answers for those "interesting questions" with Qwen and Llama models

#### 28.04 (reading monday)

- DeepSeekMath (https://arxiv.org/pdf/2402.03300)
    - main contribution is Iterative RL, GRPO and the iterative-training data accumulation techique.
    - seems kinda weird that they don't explicitly compare to PPO
    - They give three main research directions:
        - improving the reward model
        - improving the algorithms
        - improving the data collection and filtering
        
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Learning Reinforcement (https://arxiv.org/pdf/2501.12948?)

    - The authors employ large scale RL (from pre-trained and from instruction-following fine-tuned) to instill reasoning
    capability in their models. During the training process, the authors observe emergent behaviour like reflection. 
    - They use a rule-based system to evaluate the reward signal and note that they found this is the most stable and
    effective way to train systems with RL, as otherwise we'd need to maintain and iteratively train a reward model, which
    additionally may mislead the learner to exploit it via reward-hacking.
    - They did experiments with process-reward modelling but noted that they encountered difficulties, as its difficult to
    assign a special reward to any token in the reasoning chain.
    - They did experiments with MCTS guiding the generation process, but noted that the search space was simply too large
    and dividing the reasoning steps into sub-problems which then could've been solved with MCTS was hard
