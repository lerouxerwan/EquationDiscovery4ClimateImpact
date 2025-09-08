import subprocess as sp


def bash_call(command, just_print: bool=False):
    """Call bash function from python"""
    print(command)
    if just_print:
        return None
    else:
        out = sp.check_output(command, shell=True)
        if isinstance(out, bytes):
            out = out.decode("utf-8")
        return out.split('\n')[:-1]
