from azureml.core import Workspace
from azureml.core import Experiment

ws = Workspace.get("BERT", subscription_id="cdf3e529-94ee-4f54-a219-4720963fee3b")
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')
      
      
workspace=Workspace.get("BERT", subscription_id="cdf3e529-94ee-4f54-a219-4720963fee3b")
experiment_name = 'azure-service-classifier' 
experiment = Experiment(workspace, name=experiment_name)

from azureml.train.dnn import TensorFlow

from azureml.core.runconfig import RunConfiguration

# Edit a run configuration property on the fly.
run_local = RunConfiguration()

run_local.environment.python.user_managed_dependencies = True

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


estimator1 = TensorFlow(source_directory='/mnt/c/Users/phill/azure-ml/bert-stack-overflow',
                        entry_script='./1-Training/train.py',
                        compute_target="local",
                        script_params = script_params,
                        framework_version='2.0')

run1 = experiment.submit(estimator1)