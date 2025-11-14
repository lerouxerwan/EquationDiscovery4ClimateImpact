import os
import os.path as op

bash_folder = '/home/e23lerou/bash_scripts'

"""
Avant de lancer le code ci-dessous, il faut lancer avec le terminal la commande suivante:

rsync -avz -e ssh  e23lerou@sl-mee-br-101:~/shared_space/Documents/EquationDiscovery4ClimateImpact/slurm/bash_scripts/ bash_scripts/
"""

def main_jobs_status():
    prefixes = ['xp_int_1', 'xp_int_2']
    for prefix in prefixes:
        job_folders = [job_folder for job_folder in os.listdir(bash_folder) if job_folder.startswith(prefix)]
        for job_folder in job_folders[:]:
            key_sentences = ['Save fit to file', 'Load fit from file']
            c = {key_sentence: 0 for key_sentence in key_sentences}
            filepath = op.join(bash_folder, job_folder, "4294967294.out")
            for line in open(filepath, 'r').readlines():
                for key_sentence in key_sentences:
                    if key_sentence in line:
                        c[key_sentence] += 1
            print(job_folder, sum(list(c.values())), c)


if __name__ == '__main__':
    main_jobs_status()