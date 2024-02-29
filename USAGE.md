# Usage

Use Aldernet in a project:

```python
import aldernet
```

Before running, make sure to activate your project environment:

```bash
conda activate aldernet
```

Make sure to set all the necessary paths and settings in
`scr/aldernet/hyperparameters.yaml` before running it. **The training data must be
available as zarr-archives**. If you don't have access to the data, you can recreate the
zarr-archives by running the scripts in the `src/aldernet/data` folder or use the
provided zarr-archives at the default path location. To use the data at the
default path location you need access to the MeteoSwiss CSCS-Clusters. This data is not
freely available as of right now.

**THEN RUN:**

`python src/aldernet/training.py`

As the training requires lots of computational resources, it is suggested to work on a
HPC system. For example at CSCS on the Balfrin cluster you can run this command:

`srun -N1 -n1 --gres=gpu:4 --job-name=MLFlow --time=23:59:00 --partition=a100-80gb \
    --account=s83 python src/aldernet/training.py`
