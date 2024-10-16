
# Deep leaning for natural language processing
In this repository we provide the code implemenation for Group 11's project on the generalisability of machine-generated text detection across multiple domains.

Please note that the framework used in our experiment is based on the [DetectGPT paper](https://arxiv.org/abs/2301.11305v1) and the [DetectGPT repository](https://github.com/eric-mitchell/detect-gpt). The implemntation is cloned but modified to our experiments.

## Instructions
1. Download the following datasets and put them in the data folder: <br/>
[Guardian News Dataset](https://www.kaggle.com/datasets/adityakharosekar2/guardian-news-articles) <br/>
[Job posts dataset](https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom) <br/>
[Medical articles dataset](https://www.kaggle.com/datasets/chaitanyakck/medical-text/data)

2. Run main.py file for the baseline models: <br/>
`python run.py --base_model_name gpt2-medium --dataset "guardian" --scoring_model pythia-70m --mask_filling_model_name t5-small --DEVICE cuda` <br/>
Note 1: replace the scoring_model with pythia-410m or pythia-1.4b to get the baseline results for the different models. <br/>
Note 2: replace the dataset with "job_posts" or "medical" to get the baseline results for the different datasets.

3. Run finetune.py for the different models: <br/>
`python runFinetune.py --base_model_name gpt2-medium --dataset "guardian" --scoring_model pythia-70m --mask_filling_model_name t5-small --DEVICE cuda` <br/>
Note 1: replace the scoring_model with pythia-410m or pythia-1.4b to get the fine-tune results for the different models. <br/>
Note 2: replace the dataset with "job_posts" or "medical" to get the fine-tune results for the different datasets.

## Interpreting the results
Once you successfully run the script, the results will be saved in the `results/` directory. The program generates various files and here is their description:

1. **args.json** <br/>
This contains the command line arguments that were passed when the `main.py` file was run.
2. **entropy_threshold_results.json** <br/>
This file has several fields in it. The *prediction* field holds the entropy values of the real text and the sampled text. If you set *n_samples* to 200, each of the subfields (i.e ["predictions"]["real"] and  ["predictions"]["samples"]) will contain 200 values. The *raw_results* field is a list of *n_samples* dictionary where each dictionary contains the original text and its corresponding entropy value (denoted by *original_crit*), the sampled text and its corresponding entropy value (denoted by *sampled_crit*). There are some more fields towards the end of the file (for eg.  *metrics*, *pr_metrics*, etc) but I don't know what they mean.
3. **likelihood_threshold_results.json** <br/>
This file has exactly the same structure but now the values are mean log likelihoods.
4. **rank_threshold_results.json** <br/>
Same as above, but now the values are the negative rank values.
5. **logrank_threshold_results.json** <br/>
Same as above, but now the values are negative logrank values.

Finally, based on the values you select for *n_perturbation_list*, you will have more files. For eg. if you set *n_perturbation_list* to 1,10, you will have four more files:

 1. **perturbation_1_d_results.json** <br/>
	* The ["prediction"]["real"] field stores the unnormalized perturbation discrepancy for the real text (written by human) with just one (i.e *k=1*) perturbed original text to approximate the expectation term in eq 1 of the paper). The ["prediction"]["samples"] field stores the same thing but for the machine generated text. Each of the ["prediction"]["real"] and ["prediction"]["samples"] should contain *n_samples* number of entries.
	* The *raw_results* field contain *n_samples* dictionaries. Each dictionary holds the original text (*original*) and its original loglikelihood (*original_ll*), perturbed original text (*perturbed_original*; should be a list with just one string because we are only using one string to approximate the expectation term) and its loglikelihood (*all_perturbed_original_ll*; should be a list with just one entry), machine generated sample (*sampled*) and its loglikelihood (*sampled_ll*), perturbed sample (*perturbed_sampled*; again, should be a list on just one string) and its loglikelihood (*all_perturbed_sampled_ll*; again, should be a list with just one entry). 
	* The *perturbed_original_ll* holds the mean of the *all_perturbed_original_ll* list and the *perturbed_original_ll_std* holds the standard deviation. Same goes for *perturbed_sampled_ll* and *perturbed_sampled_ll_std*.

2. **perturbation_1_z_results.json** <br/>
Exactly same as above but the ["prediction"]["real"] and ["prediction"]["samples"] now store the **normalized perturbation discrepancy** values.

3. **perturbation_10_d_results.json** <br/>
Contains unnormalized perturbation discrepancy values. The only difference is that since we are using 10 perturbed samples to approximate the expectation term in eq. 1 of the paper, *perturbed_original*, *all_perturbed_original_ll*, *perturbed_sampled*, and *all_perturbed_sampled_ll* should now contain 10 values each.

4. **perturbation_10_z_results.json** <br/>
Same as above but contains **normalized perturbation discrepancy** values.
