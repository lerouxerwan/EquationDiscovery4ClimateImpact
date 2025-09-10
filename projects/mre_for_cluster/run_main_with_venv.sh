!/bin/bash
export PATH="/Odyssey/private/e23lerou/shared_install/miniforge3/bin:$PATH"
eval "$(mamba shell hook --shell bash)"
export PATH="/Odyssey/private/e23lerou/shared_install/julia-1.11.6/bin:$PATH"
export JULIA_DEPOT_PATH="/Odyssey/private/e23lerou/shared_install/.julia"
export PYTHONPATH="${PYTHONPATH}:/Odyssey/private/e23lerou/test_server/EquationDiscovery4ClimateImpact"
mamba activate /Odyssey/private/e23lerou/shared_install/venv
python projects/mre_cluster/main.py