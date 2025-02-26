import subprocess as sp


def bash_call(command, print_command=False):
    """Call bash function from python"""
    if print_command:
        print(command)
    else:
        out = sp.check_output(command, shell=True)
        if isinstance(out, bytes):
            out = out.decode("utf-8")
        return out.split('\n')[:-1]
