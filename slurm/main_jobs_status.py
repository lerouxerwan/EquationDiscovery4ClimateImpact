import os
import os.path as op

bash_folder = '/home/e23lerou/bash_scripts'

"""
Avant de lancer le code ci-dessous, il faut lancer avec le terminal la commande suivante:

rsync -avz -e ssh  e23lerou@sl-mee-br-101:~/shared_space/Documents/EquationDiscovery4ClimateImpact/slurm/bash_scripts/ bash_scripts/
"""

def main_jobs_status():
    prefixes = ['xp_int_3', 'xp_int_4'][1:]
    for prefix in prefixes:
        job_folders = [job_folder for job_folder in os.listdir(bash_folder) if job_folder.startswith(prefix)]
        for job_folder in job_folders[:]:
            key_sentences = ['finished']
            c = {key_sentence: 0 for key_sentence in key_sentences}
            filepath = op.join(bash_folder, job_folder, "4294967294.out")
            for line in open(filepath, 'r').readlines():
                for key_sentence in key_sentences:
                    if key_sentence in line:
                        c[key_sentence] += 1
            target = 210 if prefix == "xp_int_3" else 500
            percent = 100 * min(sum(list(c.values())), target) / target
            print(job_folder, f'{int(percent)}%')


"""
xp_int_4_1_2 49%
xp_int_4_1_0 49%
xp_int_4_0_1 41%
xp_int_4_0_2 43%
xp_int_4_2_1 39%
xp_int_4_2_2 57%
xp_int_4_2_0 45%
xp_int_4_1_1 44%
xp_int_4_0_0 44%
"""

"""
Le 9 Decembre
xp_int_4_1_2 91%
xp_int_4_1_0 100%
xp_int_4_0_1 85%
xp_int_4_0_2 91%
xp_int_4_2_1 100%
xp_int_4_2_2 100%
xp_int_4_2_0 91%
xp_int_4_1_1 89%
xp_int_4_0_0 87%
"""


if __name__ == '__main__':
    main_jobs_status()