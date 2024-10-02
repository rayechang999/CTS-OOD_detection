# OOD detection for artemisinin resistance

This project provides an example of using OOD detection on a real-world, medically relevant task: predicting which isolates of a malaria-causing pathogen are resistant to artemisinin, an anti-malarial drug.

The vast majority of the code is copied over from the [winning algorithm's repository](https://github.com/GuanLab/Predict-Malaria-ART-Resistance/tree/master), with light modifications to allow for validation on an *in vivo* dataset.

We provide a Jupyter notebook (`tutorial.ipynb`) with a step-by-step tutorial on using OOD detection for this task. We also provide the raw training and validation data (from [Mok. et. al](https://doi.org/doi:10.1126/science.1260403)). All the other files are from the winning algorithm's repository.

## Getting started

To get started, create a conda environment for this tutorial:

```
conda create -n my_environment_name python
conda activate my_environment_name
```

Then, install the required packages:


```
pip install -r requirements.txt
```

Now, you can start up Jupyter:

```
jupyter notebook
```


Once Jupyter starts up, you should see a list of files. Open `tutorial.ipynb` to get started! (If you have issues with the first code cell, please see **Common issues** below.)

Once you're done, you can exit Jupyter by clicking **File --> Shut down**. Then, in the command line, run `conda deactivate`.

## Common issues

**OSError: [WinError 126] The specified module could not be found. Error loading "C:\\...\\torch\\lib\\fbgemm.dll" or one of its dependencies.**

This might be an issue with installing torch through pip. Instead, try installing through conda. After activating the conda environment, run:

```
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

If the second command (`conda install ...`) doesn't work, or if you want to use a GPU, please visit the [PyTorch get started page](https://pytorch.org/get-started/locally/), and choose "Conda" as your package to get a command that works for your system.
