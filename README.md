# Aldernet: Aldernet

## Start developing

Once you created or cloned this repository, make sure the installation is running properly. Install the package dependencies with the provided script `setup_env.sh`.
Check available options with

```bash
tools/setup_env.sh -h
```
We distinguish pinned installations based on exported (reproducible) environments and free installations where the installation
is based on top-level dependencies listed in `requirements/requirements.yml`. If you start developing, you might want to do an unpinned installation and export the environment:

```bash
tools/setup_env.sh -u -e -n <package_env_name>
```

*Hint*: If you are the package administrator, it is a good idea to understand what this script does, you can do everything manually with `conda` instructions.

*Hint*: Use the flag `-m` to speed up the installation using mamba. Of course you will have to install mamba first (we recommend to install mamba into your base
environment `conda install -c conda-forge mamba`. If you install mamba in another (maybe dedicated) environment, environments installed with mamba will be located
in `<miniconda_root_dir>/envs/mamba/envs`, which is not very practical.

The package itself is installed with `pip`. For development, install in editable mode:

```bash
conda activate <package_env_name>
pip install --editable .
```

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

Once your package is installed, run the tests by typing:

```bash
conda activate <package_env_name>
pytest
```

If the tests pass, you are good to go. If not, contact the package administrator Simon Adamov. Make sure to update your requirement files and export your environments after installation
every time you add new imports while developing. Check the next section to find some guidance on the development process if you are new to Python and/or APN.

### Roadmap to your first contribution

Generally, the source code of your library is located in `src/<library_name>`. The blueprint will generate some example code in `mutable_number.py`, `utils.py` and `cli.py`. `cli.py` thereby serves as an entry
point for functionalities you want to execute from the command line, it is based on the Click library. If you do not need interactions with the command line, you should remove `cli.py`. Moreover, of course there exist other options for command line interfaces,
a good overview may be found here (<https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/>), we recommend however to use click. The provided example
code should provide some guidance on how the individual source code files interact within the library. In addition to the example code in `src/<library_name>`, there are examples for
unit tests in `tests/<library_name>/`, which can be triggered with `pytest` from the command line. Once you implemented a feature (and of course you also
implemented a meaningful test ;-)), you are likely willing to commit it. First, go to the root directory of your package and run pytest.

```bash
conda activate <package_env_name>
cd <package-root-dir>
pytest
```

If you use the tools provided by the blueprint as is, pre-commit will not be triggered locally but only if you push to the main branch
(or push to a PR to the main branch). If you consider it useful, you can set up pre-commit to run locally before every commit by initializing it once. In the root directory of
your package, type:

```bash
pre-commit install
```

If you run `pre-commit` without installing it before (line above), it will fail and the only way to recover it, is to do a forced reinstallation (`conda install --force-reinstall pre-commit`).
You can also just run pre-commit selectively, whenever you want by typing (`pre-commit run --all-files`). Note that mypy and pylint take a bit of time, so it is really
up to you, if you want to use pre-commit locally or not. In any case, after running pytest, you can commit and the linters will run at the latest on the GitHub actions server,
when you push your changes to the main branch. Note that pytest is currently not invoked by pre-commit, so it will not run automatically. Automated testing can be set up with
GitHub Actions or be implemented in a Jenkins pipeline (template for a plan available in `jenkins/`. See the next section for more details.

## Development tools

As this package was created with the APN Python blueprint, it comes with a stack of development tools, which are described in more detail on
(<https://meteoswiss-apn.github.io/mch-python-blueprint/>). Here, we give a brief overview on what is implemented.

### Testing and coding standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in APN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS
machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

### Pre-commit on GitHub actions

`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned) on the GitHub actions server and
then run various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that)
and in pull requests to the main branch.

### Jenkins

Two jenkins plans are available in the `jenkins/` folder. On the one hand `jenkins/Jenkinsfile` controls the nightly (weekly, monthly, ...) builds, on the other hand
`jenkins/JenkinsJobPR` controls the pipeline invoked with the command `launch jenkins` in pull requests on GitHub. Your jenkins pipeline will not be set up
automatically. If you need to run your tests on CSCS machines, contact DevOps to help you with the setup of the pipelines. Otherwise, you can ignore the jenkinsfiles
and exclusively run your tests and checks on GitHub actions.

## Features

This repo contains the code to train the Aldernet neural network model to predict surface level pollen concentrations.
To retrain the model simply run this command from the project root directory:

`python src/aldernet/training.py`

As the training requires lots of computational resources, it is suggested to work on a HPC system.
For example at CSCS on the Balfrin cluster you can run this command:

`srun -N1 -n1 --gres=gpu:4 --job-name=MLFlow --time=23:59:00 --partition=normal --account=s83 python src/aldernet/training.py`

The training is conducted by ray tune for parallel computations and hyper-parameter tuning and
MLFlow for logging and checkpointing.

Define training setting on lines 39-52 in the file `src/aldernet/training.py`
and paths to the input data on lines 54-79, before starting the training.
To use the data at the default path location you need access to the MeteoSwiss CSCS-Clusters.
This data is not freely available as of right now.

The following is a list of the most important files in this repo with a short explanation:

- **data**:
  - `fieldextra_alnu.nl/fieldextra_cory.nl`: [`Fieldextra`](https://github.com/COSMO-ORG/fieldextra) namelist to retrieve Cosmo model output data
  - `retrieve_dwh.sh`: retrieval of pollen station measurements from the MeteoSwiss Data-Warehouse
  - `scaling.txt`: the scaling applied to the Alder pollen concentrations before model training
  - `species.RData/stations.RData`: lists containing names and abbreviations of pollen species and stations
- **notebooks**:
  - `analysis.Rmd`: statistical verification of modelled vs. measured concentrations at station level
  - `example_cosmo_pollen.ipynp`: mapplot using Psyplot and Iconarray packages
  - `profiling.ipynp`: descriptive statistics of input features and their correlations
- **src/aldernet:**
  - `training.py`: the main script to start the training of the neural network
  - `utils.py`: containing all functions required by the training.py script
  - `plots.py`: creation of mapplots and animated gifs
  - **data:**
    - `data_202X.py`: yearly aggregation of GRIB2 model output into zarr archives
    - `rechunk_zarr.py`: combined all yearly zarr archives into one and rechunk by valid_time
    - `data_utils.py`: various functions required by the other scripts in the data folder
    - `create_batcher_input.np:` Preprocessing and train/test split of the data input

## Credits

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://meteoswiss-apn.github.io/mch-python-blueprint/) project template.
