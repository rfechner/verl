"""
    - CLI script to delegate to shell scripts running rollouts/analysis.

    - Running rollouts:
        - for every checkpoint in a directory start a rollout job.
        - make sure to collect everything necessary -> advantages, entropy, rewards etc.
    
    - Running log-likelihoods:
        - for every checkpoint in a directory start (something like) a rollout job
            - will have to write a script which just collects log-likelihoods.
            -> Actor wg has function compute_log_likelihoods or something like that.
"""
import os

def main():
    pass