# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.
# %% [markdown]
# # Part 1: Training Tensorflow 2.0 Model on Azure Machine Learning Service
# 
# ## Overview of the part 1
# This notebook is Part 1 (Preparing Data and Model Training) of a four part workshop that demonstrates an end-to-end workflow using Tensorflow 2.0 on Azure Machine Learning service. The different components of the workshop are as follows:
# 
# - Part 1: [Preparing Data and Model Training](https://github.com/microsoft/bert-stack-overflow/blob/master/1-Training/AzureServiceClassifier_Training.ipynb)
# - Part 2: [Inferencing and Deploying a Model](https://github.com/microsoft/bert-stack-overflow/blob/master/2-Inferencing/AzureServiceClassifier_Inferencing.ipynb)
# - Part 3: [Setting Up a Pipeline Using MLOps](https://github.com/microsoft/bert-stack-overflow/tree/master/3-ML-Ops)
# - Part 4: [Explaining Your Model Interpretability](https://github.com/microsoft/bert-stack-overflow/blob/master/4-Interpretibility/IBMEmployeeAttritionClassifier_Interpretability.ipynb)
# 
# **This notebook will cover the following topics:**
# 
# - Stackoverflow question tagging problem
# - Introduction to Transformer and BERT deep learning models
# - Introduction to Azure Machine Learning service
# - Preparing raw data for training using Apache Spark
# - Registering cleaned up training data as a Dataset
# - Debugging the model in Tensorflow 2.0 Eager Mode
# - Training the model on GPU cluster
# - Monitoring training progress with built-in Tensorboard dashboard 
# - Automated search of best hyper-parameters of the model
# - Registering the trained model for future deployment
# %% [markdown]
# ## Prerequisites
# This notebook is designed to be run in Azure ML Notebook VM. See [readme](https://github.com/microsoft/bert-stack-overflow/blob/master/README.md) file for instructions on how to create Notebook VM and open this notebook in it.
# %% [markdown]
# ### Check Azure Machine Learning Python SDK version
# 
# This tutorial requires version 1.0.69 or higher. Let's check the version of the SDK:

# %%
import azureml.core

print("Azure Machine Learning Python SDK version:", azureml.core.VERSION)

# %% [markdown]
# ## Stackoverflow Question Tagging Problem 
# In this workshop we will use powerful language understanding model to automatically route Stackoverflow questions to the appropriate support team on the example of Azure services.
# 
# One of the key tasks to ensuring long term success of any Azure service is actively responding to related posts in online forums such as Stackoverflow. In order to keep track of these posts, Microsoft relies on the associated tags to direct questions to the appropriate support team. While Stackoverflow has different tags for each Azure service (azure-web-app-service, azure-virtual-machine-service, etc), people often use the generic **azure** tag. This makes it hard for specific teams to track down issues related to their product and as a result, many questions get left unanswered. 
# 
# **In order to solve this problem, we will build a model to classify posts on Stackoverflow with the appropriate Azure service tag.**
# 
# We will be using a BERT (Bidirectional Encoder Representations from Transformers) model which was published by researchers at Google AI Reasearch. Unlike prior language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of natural language processing (NLP) tasks without substantial architecture modifications.
# 
# ## Why use BERT model?
# [Introduction of BERT model](https://arxiv.org/pdf/1810.04805.pdf) changed the world of NLP. Many NLP problems that before relied on specialized models to achive state of the art performance are now solved with BERT better and with more generic approach.
# 
# If we look at the leaderboards on such popular NLP problems as GLUE and SQUAD, most of the top models are based on BERT:
# * [GLUE Benchmark Leaderboard](https://gluebenchmark.com/leaderboard/)
# * [SQuAD Benchmark Leaderboard](https://rajpurkar.github.io/SQuAD-explorer/)
# 
# Recently, Allen Institue for AI announced new language understanding system called Aristo [https://allenai.org/aristo/](https://allenai.org/aristo/). The system has been developed for 20 years, but it's performance was stuck at 60% on 8th grade science test. The result jumped to 90% once researchers adopted BERT as core language understanding component. With BERT Aristo now solves the test with A grade.  
# %% [markdown]
# ## Quick Overview of How BERT model works
# 
# The foundation of BERT model is Transformer model, which was introduced in [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762). Before that event the dominant way of processing language was Recurrent Neural Networks (RNNs). Let's start our overview with RNNs.
# 
# ## RNNs
# 
# RNNs were powerful way of processing language due to their ability to memorize its previous state and perform sophisticated inference based on that.
# 
# <img src="https://miro.medium.com/max/400/1*L38xfe59H5tAgvuIjKoWPg.png" alt="Drawing" style="width: 100px;"/>
# 
# _Taken from [1](https://towardsdatascience.com/transformers-141e32e69591)_
# 
# Applied to language translation task, the processing dynamics looked like this.
# 
# ![](https://miro.medium.com/max/1200/1*8GcdjBU5TAP36itWBcZ6iA.gif)
# _Taken from [2](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)_
#     
# But RNNs suffered from 2 disadvantes:
# 1. Sequential computation put a limit on parallelization, which limited effectiveness of larger models.
# 2. Long term relationships between words were harder to detect.
# %% [markdown]
# ## Transformers
# 
# Transformers were designed to address these two limitations of RNNs.
# 
# <img src="https://miro.medium.com/max/2436/1*V2435M1u0tiSOz4nRBfl4g.png" alt="Drawing" style="width: 500px;"/>
# 
# _Taken from [3](http://jalammar.github.io/illustrated-transformer/)_
# 
# In each Encoder layer Transformer performs Self-Attention operation which detects relationships between all word embeddings in one matrix multiplication operation. 
# 
# <img src="https://miro.medium.com/max/2176/1*fL8arkEFVKA3_A7VBgapKA.gif" alt="Drawing" style="width: 500px;"/>
# 
# _Taken from [4](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)_
# 
# %% [markdown]
# ## BERT Model
# 
# BERT is a very large network with multiple layers of Transformers (12 for BERT-base, and 24 for BERT-large). The model is first pre-trained on large corpus of text data (WikiPedia + books) using un-superwised training (predicting masked words in a sentence). During pre-training the model absorbs significant level of language understanding.
# 
# <img src="http://jalammar.github.io/images/bert-output-vector.png" alt="Drawing" style="width: 700px;"/>
# 
# _Taken from [5](http://jalammar.github.io/illustrated-bert/)_
# 
# Pre-trained network then can easily be fine-tuned to solve specific language task, like answering questions, or categorizing spam emails.
# 
# <img src="http://jalammar.github.io/images/bert-classifier.png" alt="Drawing" style="width: 700px;"/>
# 
# _Taken from [5](http://jalammar.github.io/illustrated-bert/)_
# 
# The end-to-end training process of the stackoverflow question tagging model looks like this:
# 
# ![](images/model-training-e2e.png)
# 
# %% [markdown]
# ## What is Azure Machine Learning Service?
# Azure Machine Learning service is a cloud service that you can use to develop and deploy machine learning models. Using Azure Machine Learning service, you can track your models as you build, train, deploy, and manage them, all at the broad scale that the cloud provides.
# ![](./images/aml-overview.png)
# 
# 
# #### How can we use it for training machine learning models?
# Training machine learning models, particularly deep neural networks, is often a time- and compute-intensive task. Once you've finished writing your training script and running on a small subset of data on your local machine, you will likely want to scale up your workload.
# 
# To facilitate training, the Azure Machine Learning Python SDK provides a high-level abstraction, the estimator class, which allows users to easily train their models in the Azure ecosystem. You can create and use an Estimator object to submit any training code you want to run on remote compute, whether it's a single-node run or distributed training across a GPU cluster.
# %% [markdown]
# ## Connect To Workspace
# 
# The [workspace](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace(class)?view=azure-ml-py) is the top-level resource for Azure Machine Learning, providing a centralized place to work with all the artifacts you create when you use Azure Machine Learning. The workspace holds all your experiments, compute targets, models, datastores, etc.
# 
# You can [open ml.azure.com](https://ml.azure.com) to access your workspace resources through a graphical user interface of **Azure Machine Learning studio**.
# 
# ![](./images/aml-workspace.png)
# 
# **You will be asked to login in the next step. Use your Microsoft AAD credentials.**

# %%
from azureml.core import Workspace

workspace = Workspace.from_config()
print('Workspace name: ' + workspace.name, 
      'Azure region: ' + workspace.location, 
      'Subscription id: ' + workspace.subscription_id, 
      'Resource group: ' + workspace.resource_group, sep = '\n')

# %% [markdown]
# ## Create Compute Target
# 
# A [compute target](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.computetarget?view=azure-ml-py) is a designated compute resource/environment where you run your training script or host your service deployment. This location may be your local machine or a cloud-based compute resource. Compute targets can be reused across the workspace for different runs and experiments. 
# 
# For this tutorial, we will create an auto-scaling [Azure Machine Learning Compute](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.amlcompute?view=azure-ml-py) cluster, which is a managed-compute infrastructure that allows the user to easily create a single or multi-node compute. To create the cluster, we need to specify the following parameters:
# 
# - `vm_size`: The is the type of GPUs that we want to use in our cluster. For this tutorial, we will use **Standard_NC12s_v3 (NVIDIA V100) GPU Machines** .
# - `idle_seconds_before_scaledown`: This is the number of seconds before a node will scale down in our auto-scaling cluster. We will set this to **6000** seconds. 
# - `min_nodes`: This is the minimum numbers of nodes that the cluster will have. To avoid paying for compute while they are not being used, we will set this to **0** nodes.
# - `max_modes`: This is the maximum number of nodes that the cluster will scale up to. Will will set this to **2** nodes.
# 
# **When jobs are submitted to the cluster it takes approximately 5 minutes to allocate new nodes** 

# %%
from azureml.core.compute import AmlCompute, ComputeTarget

cluster_name = 'v100cluster'
compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC12s_v3', 
                                                       idle_seconds_before_scaledown=6000,
                                                       min_nodes=0, 
                                                       max_nodes=2)

compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True)

# %% [markdown]
# To ensure our compute target was created successfully, we can check it's status.

# %%
compute_target.get_status().serialize()

# %% [markdown]
# #### If the compute target has already been created, then you (and other users in your workspace) can directly run this cell.

# %%
compute_target = workspace.compute_targets['v100cluster']

# %% [markdown]
# ## Prepare Data Using Apache Spark
# 
# To train our model, we used the Stackoverflow data dump from [Stack exchange archive](https://archive.org/download/stackexchange). Since the Stackoverflow _posts_ dataset is 12GB, we prepared the data using [Apache Spark](https://spark.apache.org/) framework on a scalable Spark compute cluster in [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/). 
# 
# For the purpose of this tutorial, we have processed the data ahead of time and uploaded it to an [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/) container. The full data processing notebook can be found in the _spark_ folder.
# 
# * **ACTION**: Open and explore [data preparation notebook](spark/stackoverflow-data-prep.ipynb).
# 
# %% [markdown]
# ## Register Datastore
# %% [markdown]
# A [Datastore](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.datastore.datastore?view=azure-ml-py) is used to store connection information to a central data storage. This allows you to access your storage without having to hard code this (potentially confidential) information into your scripts. 
# 
# In this tutorial, the data was been previously prepped and uploaded into a central [Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/) container. We will register this container into our workspace as a datastore using a [shared access signature (SAS) token](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview). 

# %%
from azureml.core import Datastore, Dataset

datastore_name = 'tfworld'
container_name = 'azureml-blobstore-7c6bdd88-21fa-453a-9c80-16998f02935f'
account_name = 'tfworld6818510241'
sas_token = '?sv=2019-02-02&ss=bfqt&srt=sco&sp=rl&se=2020-06-01T14:18:31Z&st=2019-11-05T07:18:31Z&spr=https&sig=Z4JmM0V%2FQzoFNlWS3a3vJxoGAx58iCz2HAWtmeLDbGE%3D'

datastore = Datastore.register_azure_blob_container(workspace=workspace, 
                                                    datastore_name=datastore_name, 
                                                    container_name=container_name,
                                                    account_name=account_name, 
                                                    sas_token=sas_token)

# %% [markdown]
# #### If the datastore has already been registered, then you (and other users in your workspace) can directly run this cell.

# %%
datastore = workspace.datastores['tfworld']

# %% [markdown]
# #### What if my data wasn't already hosted remotely?
# All workspaces also come with a blob container which is registered as a default datastore. This allows you to easily upload your own data to a remote storage location. You can access this datastore and upload files as follows:
# ```
# datastore = workspace.get_default_datastore()
# ds.upload(src_dir='<LOCAL-PATH>', target_path='<REMOTE-PATH>')
# ```
# 
# %% [markdown]
# ## Register Dataset
# 
# Azure Machine Learning service supports first class notion of a Dataset. A [Dataset](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py) is a resource for exploring, transforming and managing data in Azure Machine Learning. The following Dataset types are supported:
# 
# * [TabularDataset](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset?view=azure-ml-py) represents data in a tabular format created by parsing the provided file or list of files.
# 
# * [FileDataset](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.filedataset?view=azure-ml-py) references single or multiple files in datastores or from public URLs.
# 
# First, we will use visual tools in Azure ML studio to register and explore our dataset as Tabular Dataset.
# 
# * **ACTION**: Follow [create-dataset](images/create-dataset.ipynb) guide to create Tabular Dataset from our training data.
# %% [markdown]
# #### Use created dataset in code

# %%
from azureml.core import Dataset

# Get a dataset by name
tabular_ds = Dataset.get_by_name(workspace=workspace, name='Stackoverflow dataset')

# Load a TabularDataset into pandas DataFrame
df = tabular_ds.to_pandas_dataframe()

df.head(10)

# %% [markdown]
# ## Register Dataset using SDK
# 
# In addition to UI we can register datasets using SDK. In this workshop we will register second type of Datasets using code - File Dataset. File Dataset allows specific folder in our datastore that contains our data files to be registered as a Dataset.
# 
# There is a folder within our datastore called **azure-service-data** that contains all our training and testing data. We will register this as a dataset.

# %%
azure_dataset = Dataset.File.from_files(path=(datastore, 'azure-service-classifier/data'))

azure_dataset = azure_dataset.register(workspace=workspace,
                                       name='Azure Services Dataset',
                                       description='Dataset containing azure related posts on Stackoverflow')

# %% [markdown]
# #### If the dataset has already been registered, then you (and other users in your workspace) can directly run this cell.

# %%
azure_dataset = workspace.datasets['Azure Services Dataset']

# %% [markdown]
# ## Explore Training Code
# %% [markdown]
# In this workshop the training code is provided in [train.py](./train.py) and [model.py](./model.py) files. The model is based on popular [huggingface/transformers](https://github.com/huggingface/transformers) libary. Transformers library provides performant implementation of BERT model with high level and easy to use APIs based on Tensorflow 2.0.
# 
# ![](https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png)
# 
# *  **ACTION**: Explore _train.py_ and _model.py_ using [Azure ML studio > Notebooks tab](images/azuremlstudio-notebooks-explore.png)
# * NOTE: You can also explore the files using Jupyter or Jupyter Lab UI.
# %% [markdown]
# ## Test Locally
# 
# Let's try running the script locally to make sure it works before scaling up to use our compute cluster. To do so, you will need to install the transformers libary.

# %%
get_ipython().run_line_magic('pip', 'install transformers==2.0.0')

# %% [markdown]
# We have taken a small partition of the dataset and included it in this repository. Let's take a quick look at the format of the data.

# %%
data_dir = 'C:/Users/phill/azure-ml/bert-stack-overflow/1-Training/data-shared-inbox/'


# %%
import os 
import pandas as pd
data = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
data.head(5)

# %% [markdown]
# Now we know what the data looks like, let's test out our script!

# %%
import sys
get_ipython().system('{sys.executable} train.py --data_dir {data_dir} --max_seq_length 128 --batch_size 16 --learning_rate 3e-5 --steps_per_epoch 5 --num_epochs 1 --export_dir ../outputs/model')

# %% [markdown]
# ## Debugging in TensorFlow 2.0 Eager Mode
# 
# Eager mode is new feature in TensorFlow 2.0 which makes understanding and debugging models easy. Let's start by configuring our remote debugging environment.
# 
# #### Configure VS Code Remote connection to Notebook VM
# 
# * **ACTION**: Install [Microsoft VS Code](https://code.visualstudio.com/) on your local machine.
# 
# * **ACTION**: Follow this [configuration guide](https://github.com/danielsc/azureml-debug-training/blob/master/Setting%20up%20VSCode%20Remote%20on%20an%20AzureML%20Notebook%20VM.md) to setup VS Code Remote connection to Notebook VM.
# 
# #### Debug training code using step-by-step debugger
# 
# * **ACTION**: Open Remote VS Code session to your Notebook VM.
# * **ACTION**: Open file `/home/azureuser/cloudfiles/code/<username>/bert-stack-overflow/1-Training/train_eager.py`.
# * **ACTION**: Set break point in the file and start Python debugging session. 
# 
# %% [markdown]
# On a CPU machine training on a full dataset will take approximatly 1.5 hours. Although it's a small dataset, it still takes a long time. Let's see how we can speed up the training by using latest NVidia V100 GPUs in the Azure cloud. 
# %% [markdown]
# ## Perform Experiment
# 
# Now that we have our compute target, dataset, and training script working locally, it is time to scale up so that the script can run faster. We will start by creating an [experiment](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.experiment?view=azure-ml-py). An experiment is a grouping of many runs from a specified script. All runs in this tutorial will be performed under the same experiment. 

# %%

from azureml.core import Workspace

ws = Workspace.get("BERT", subscription_id="cdf3e529-94ee-4f54-a219-4720963fee3b")
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')


# %%
from azureml.core import Experiment
workspace=Workspace.get("BERT", subscription_id="cdf3e529-94ee-4f54-a219-4720963fee3b")
experiment_name = 'azure-service-classifier' 
experiment = Experiment(workspace, name=experiment_name)
experiment

# %% [markdown]
# #### Create TensorFlow Estimator
# 
# The Azure Machine Learning Python SDK Estimator classes allow you to easily construct run configurations for your experiments. They allow you too define parameters such as the training script to run, the compute target to run it on, framework versions, additional package requirements, etc. 
# 
# You can also use a generic [Estimator](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator.estimator?view=azure-ml-py) to submit training scripts that use any learning framework you choose.
# 
# For popular libaries like PyTorch and Tensorflow you can use their framework specific estimators. We will use the [TensorFlow Estimator](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.dnn.tensorflow?view=azure-ml-py) for our experiment.

# %%
from azureml.train.dnn import TensorFlow

script_params = {
    # to mount files referenced by mnist dataset
    '--data_dir': './1-Training/data-shared-inbox',
    '--max_seq_length': 128,
    '--batch_size': 32,
    '--learning_rate': 3e-5,
    '--steps_per_epoch': 150,
    '--num_epochs': 3,
    '--export_dir':'./1-Training/outputs/model'
}

estimator1 = TensorFlow(source_directory='C:/Users/phill/azure-ml/bert-stack-overflow',
                        entry_script='./1-Training/train.py',
                        compute_target="local",
                        script_params = script_params,
                        framework_version='2.0')

# %% [markdown]
# A quick description for each of the parameters we have just defined:
# 
# - `source_directory`: This specifies the root directory of our source code. 
# - `entry_script`: This specifies the training script to run. It should be relative to the source_directory.
# - `compute_target`: This specifies to compute target to run the job on. We will use the one created earlier.
# - `script_params`: This specifies the input parameters to the training script. Please note:
# 
#     1) *azure_dataset.as_named_input('azureservicedata').as_mount()* mounts the dataset to the remote compute and provides the path to the dataset on our datastore. 
#     
#     2) All outputs from the training script must be outputted to an './outputs' directory as this is the only directory that will be saved to the run. 
#     
#     
# - `framework_version`: This specifies the version of TensorFlow to use. Use Tensorflow.get_supported_verions() to see all supported versions.
# - `use_gpu`: This will use the GPU on the compute target for training if set to True.
# - `pip_packages`: This allows you to define any additional libraries to install before training.
# %% [markdown]
# #### 1) Submit First Run 
# 
# We can now train our model by submitting the estimator object as a [run](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py).

# %%
run1 = experiment.submit(estimator1)

# %% [markdown]
# We can view the current status of the run and stream the logs from within the notebook.

# %%
from azureml.widgets import RunDetails
RunDetails(run1).show()

# %% [markdown]
# You cancel a run at anytime which will stop the run and scale down the nodes in the compute target.

# %%
run1.cancel()

# %% [markdown]
# While we wait for the run to complete, let's go over how a Run is executed in Azure Machine Learning.
# 
# ![](./images/aml-run.png)
# %% [markdown]
# #### 2) Add Metrics Logging
# 
# So we were able to clone a Tensorflow 2.0 project and run it without any changes. However, with larger scale projects we would want to log some metrics in order to make it easier to monitor the performance of our model. 
# 
# We can do this by adding a few lines of code into our training script:
# 
# ```python
# # 1) Import SDK Run object
# from azureml.core.run import Run
# 
# # 2) Get current service context
# run = Run.get_context()
# 
# # 3) Log the metrics that we want
# run.log('val_accuracy', float(logs.get('val_accuracy')))
# run.log('accuracy', float(logs.get('accuracy')))
# ```
# We've created a *train_logging.py* script that includes logging metrics as shown above. 
# 
# *  **ACTION**: Explore _train_logging.py_ using [Azure ML studio > Notebooks tab](images/azuremlstudio-notebooks-explore.png)
# %% [markdown]
# We can submit this run in the same way that we did before. 
# 
# *Since our cluster can scale automatically to two nodes, we can run this job simultaneously with the previous one.*

# %%
script_params = {
    # to mount files referenced by mnist dataset
    '--data_dir': './1-Training/data-shared-inbox',
    '--max_seq_length': 128,
    '--batch_size': 32,
    '--learning_rate': 3e-5,
    '--steps_per_epoch': 150,
    '--num_epochs': 3,
    '--export_dir':'./1-Training/outputs/model'
}

estimator2 = TensorFlow(source_directory='C:/Users/phill/azure-ml/bert-stack-overflow',
                        entry_script='./1-Training/train_logging.py',
                        compute_target="local",
                        script_params = script_params,
                        framework_version='2.0')

run2 = experiment.submit(estimator2)

# %% [markdown]
# Now if we view the current details of the run, you will notice that the metrics will be logged into graphs.

# %%
from azureml.widgets import RunDetails
RunDetails(run2).show()

# %% [markdown]
# #### 3) Monitoring metrics with Tensorboard
# 
# Tensorboard is a popular Deep Learning Training visualization tool and it's built-in into TensorFlow framework. We can easily add tracking of the metrics in Tensorboard format by adding Tensorboard callback to the **fit** function call.
# ```python
#     # Add callback to record Tensorboard events
#     model.fit(train_dataset, epochs=FLAGS.num_epochs, 
#               steps_per_epoch=FLAGS.steps_per_epoch, validation_data=valid_dataset, 
#               callbacks=[
#                   AmlLogger(),
#                   tf.keras.callbacks.TensorBoard(update_freq='batch')]
#              )
# ```
# 
# #### Launch Tensorboard
# Azure ML service provides built-in integration with Tensorboard through **tensorboard** package.
# 
# While the run is in progress (or after it has completed), we can start Tensorboard with the run as its target, and it will begin streaming logs.

# %%
from azureml.tensorboard import Tensorboard

# The Tensorboard constructor takes an array of runs, so be sure and pass it in as a single-element array here
tb = Tensorboard([run2])

# If successful, start() returns a string with the URI of the instance.
tb.start()

# %% [markdown]
# #### Stop Tensorboard
# When you're done, make sure to call the stop() method of the Tensorboard object, or it will stay running even after your job completes.

# %%
tb.stop()

# %% [markdown]
# ## Check the model performance
# 
# Last training run produced model of decent accuracy. Let's test it out and see what it does. First, let's check what files our latest training run produced and download the model files.
# 
# #### Download model files

# %%
run2.get_file_names()


# %%
run2.download_files(prefix='outputs/model')

# If you haven't finished training the model then just download pre-made model from datastore
datastore.download('./',prefix="azure-service-classifier/model")

# %% [markdown]
# #### Instantiate the model
# 
# Next step is to import our model class and instantiate fine-tuned model from the model file.

# %%
from model import TFBertForMultiClassification
from transformers import BertTokenizer
import tensorflow as tf
def encode_example(text, max_seq_length):
    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length
        )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    return input_ids, attention_mask, token_type_ids
    
labels = ['azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions']
# Load model and tokenizer
loaded_model = TFBertForMultiClassification.from_pretrained('azure-service-classifier/model', num_labels=len(labels))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Model loaded from disk.")

# %% [markdown]
# #### Define prediction function
# 
# Using the model object we can interpret new questions and predict what Azure service they talk about. To do that conveniently we'll define **predict** function.

# %%
# Prediction function
def predict(question):
    input_ids, attention_mask, token_type_ids = encode_example(question, 128)
    predictions = loaded_model.predict({
        'input_ids': tf.convert_to_tensor([input_ids], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([attention_mask], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([token_type_ids], dtype=tf.int32)
    })
    prediction = labels[predictions[0].argmax().item()]
    probability = predictions[0].max()
    result = {
        'prediction': str(labels[predictions[0].argmax().item()]),
        'probability': str(predictions[0].max())
    }
    print('Prediction: {}'.format(prediction))
    print('Probability: {}'.format(probability))

# %% [markdown]
# #### Experiement with our new model
# 
# Now we can easily test responses of the model to new inputs. 
# *  **ACTION**: Invent yout own input for one of the 5 services our model understands: 'azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions'.

# %%
# Route question
predict("How can I specify Service Principal in devops pipeline when deploying virtual machine")


# %%
# Now more tricky cae - the opposite
predict("How can virtual machine trigger devops pipeline")

# %% [markdown]
# ## Distributed Training Across Multiple GPUs
# 
# Distributed training allows us to train across multiple nodes if your cluster allows it. Azure Machine Learning service helps manage the infrastructure for training distributed jobs. All we have to do is add the following parameters to our estimator object in order to enable this:
# 
# - `node_count`: The number of nodes to run this job across. Our cluster has a maximum node limit of 2, so we can set this number up to 2.
# - `process_count_per_node`: The number of processes to enable per node. The nodes in our cluster have 2 GPUs each. We will set this value to 2 which will allow us to distribute the load on both GPUs. Using multi-GPUs nodes is benefitial as communication channel bandwidth on local machine is higher.
# - `distributed_training`: The backend to use for our distributed job. We will be using an MPI (Message Passing Interface) backend which is used by Horovod framework.
# 
# We use [Horovod](https://github.com/horovod/horovod), which is a framework that allows us to easily modifying our existing training script to be run across multiple nodes/GPUs. The distributed training script is saved as *train_horovod.py*.
# 
# *  **ACTION**: Explore _train_horovod.py_ using [Azure ML studio > Notebooks tab](images/azuremlstudio-notebooks-explore.png)
# %% [markdown]
# We can submit this run in the same way that we did with the others, but with the additional parameters.

# %%
from azureml.train.dnn import Mpi

estimator3 = TensorFlow(source_directory='./',
                        entry_script='train_horovod.py',compute_target=compute_target,
                        script_params = {
                              '--data_dir': azure_dataset.as_named_input('azureservicedata').as_mount(),
                              '--max_seq_length': 128,
                              '--batch_size': 32,
                              '--learning_rate': 3e-5,
                              '--steps_per_epoch': 150,
                              '--num_epochs': 3,
                              '--export_dir':'./outputs/model'
                        },
                        framework_version='2.0',
                        node_count=1,
                        distributed_training=Mpi(process_count_per_node=2),
                        use_gpu=True,
                        pip_packages=['transformers==2.0.0', 'azureml-dataprep[fuse,pandas]==1.1.29'])

run3 = experiment.submit(estimator3)

# %% [markdown]
# Once again, we can view the current details of the run. 

# %%
from azureml.widgets import RunDetails
RunDetails(run3).show()

# %% [markdown]
# Once the run completes note the time it took. It should be around 5 minutes. As you can see, by moving to the cloud GPUs and using distibuted training we managed to reduce training time of our model from more than an hour to 5 minutes. This greatly improves speed of experimentation and innovation.
# %% [markdown]
# ## Tune Hyperparameters Using Hyperdrive
# 
# So far we have been putting in default hyperparameter values, but in practice we would need tune these values to optimize the performance. Azure Machine Learning service provides many methods for tuning hyperparameters using different strategies.
# 
# The first step is to choose the parameter space that we want to search. We have a few choices to make here :
# 
# - **Parameter Sampling Method**: This is how we select the combinations of parameters to sample. Azure Machine Learning service offers [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py), [GridParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.gridparametersampling?view=azure-ml-py), and [BayesianParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.bayesianparametersampling?view=azure-ml-py). We will use the `GridParameterSampling` method.
# - **Parameters To Search**: We will be searching for optimal combinations of `learning_rate` and `num_epochs`.
# - **Parameter Expressions**: This defines the [functions that can be used to describe a hyperparameter search space](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py), which can be discrete or continuous. We will be using a `discrete set of choices`.
# 
# The following code allows us to define these options.

# %%
from azureml.train.hyperdrive import GridParameterSampling
from azureml.train.hyperdrive.parameter_expressions import choice


param_sampling = GridParameterSampling( {
        '--learning_rate': choice(3e-5, 3e-4),
        '--num_epochs': choice(3, 4)
    }
)

# %% [markdown]
# The next step is to a define how we want to measure our performance. We do so by specifying two classes:
# 
# - **[PrimaryMetricGoal](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.primarymetricgoal?view=azure-ml-py)**: We want to `MAXIMIZE` the `val_accuracy` that is logged in our training script.
# - **[BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)**: A policy for early termination so that jobs which don't show promising results will stop automatically.

# %%
from azureml.train.hyperdrive import BanditPolicy
from azureml.train.hyperdrive import PrimaryMetricGoal

primary_metric_name='val_accuracy'
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE

early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=2)

# %% [markdown]
# We define an estimator as usual, but this time without the script parameters that we are planning to search.

# %%
estimator4 = TensorFlow(source_directory='./',
                        entry_script='train_logging.py',
                        compute_target=compute_target,
                        script_params = {
                              '--data_dir': azure_dataset.as_named_input('azureservicedata').as_mount(),
                              '--max_seq_length': 128,
                              '--batch_size': 32,
                              '--steps_per_epoch': 150,
                              '--export_dir':'./outputs/model',
                        },
                        framework_version='2.0',
                        use_gpu=True,
                        pip_packages=['transformers==2.0.0', 'azureml-dataprep[fuse,pandas]==1.1.29'])

# %% [markdown]
# Finally, we add all our parameters in a [HyperDriveConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py) class and submit it as a run. 

# %%
from azureml.train.hyperdrive import HyperDriveConfig

hyperdrive_run_config = HyperDriveConfig(estimator=estimator4,
                                         hyperparameter_sampling=param_sampling, 
                                         policy=early_termination_policy,
                                         primary_metric_name=primary_metric_name, 
                                         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                         max_total_runs=10,
                                         max_concurrent_runs=2)

run4 = experiment.submit(hyperdrive_run_config)

# %% [markdown]
# When we view the details of our run this time, we will see information and metrics for every run in our hyperparameter tuning.

# %%
from azureml.widgets import RunDetails
RunDetails(run4).show()

# %% [markdown]
# We can retrieve the best run based on our defined metric.

# %%
best_run = run4.get_best_run_by_primary_metric()

# %% [markdown]
# ## Register Model
# 
# A registered [model](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model(class)?view=azure-ml-py) is a reference to the directory or file that make up your model. After registering a model, you and other people in your workspace can easily gain access to and deploy your model without having to run the training script again. 
# 
# We need to define the following parameters to register a model:
# 
# - `model_name`: The name for your model. If the model name already exists in the workspace, it will create a new version for the model.
# - `model_path`: The path to where the model is stored. In our case, this was the *export_dir* defined in our estimators.
# - `description`: A description for the model.
# 
# Let's register the best run from our hyperparameter tuning.

# %%
model = best_run.register_model(model_name='azure-service-classifier', 
                                model_path='./outputs/model',
                                datasets=[('train, test, validation data', azure_dataset)],
                                description='BERT model for classifying azure services on stackoverflow posts.')

# %% [markdown]
# We have registered the model with Dataset reference. 
# * **ACTION**: Check dataset to model link in **Azure ML studio > Datasets tab > Azure Service Dataset**.
# %% [markdown]
# In the [next tutorial](), we will perform inferencing on this model and deploy it to a web service.

