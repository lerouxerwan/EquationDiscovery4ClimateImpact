!/bin/bash
eval "$(/Odyssey/private/e23lerou/shared_install/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate /Odyssey/private/e23lerou/shared_install/venv
export PATH="/Odyssey/private/e23lerou/shared_install/julia-1.11.6/bin:$PATH"
export JULIA_DEPOT_PATH="/Odyssey/private/e23lerou/shared_install/.julia"
export PYTHONPATH="${PYTHONPATH}:/Odyssey/private/e23lerou/test_server/EquationDiscovery4ClimateImpact"
python projects/mre_for_cluster/main.py
