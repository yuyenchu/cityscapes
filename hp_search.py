from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.hpbandster import OptimizerBOHB

from clearml import Task

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('Record Broke! Objective reached {}'.format(objective_value))


task = Task.init(project_name='Hyperparameter Optimization with BOHB',
                 task_name='Hyperparameter Search: semantic_segmentation',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)
configs = {'template_task_id': '689bb494cfc14029b81ed29fee8ca480'}
configs = task.connect(configs)

optimizer = HyperParameterOptimizer(
    base_task_id=configs['template_task_id'],  # experiment to optimize
    # hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('Args/epochs', min_value=10, max_value=100, step_size=2),
        UniformIntegerParameterRange('Args/kfold', min_value=5, max_value=20, step_size=1),
        UniformIntegerParameterRange('Args/batch_size', min_value=8, max_value=20, step_size=4),
        UniformParameterRange('Args/initial_lr', min_value=0.00025, max_value=0.005, step_size=0.00025),
        UniformIntegerParameterRange('Args/first_decay_epoch', min_value=2, max_value=15, step_size=1),
        UniformParameterRange('Args/m_mul', min_value=0.1, max_value=0.9, step_size=0.05),
        UniformParameterRange('Args/alpha', min_value=1, max_value=3, step_size=0.25),
        UniformParameterRange('Args/delta', min_value=0.001, max_value=0.2, step_size=0.001),
        UniformParameterRange('Args/lambda_val', min_value=0, max_value=0.3, step_size=0.0005),
    ],
    # objective metric
    objective_metric_title='epoch_softmax_out_sparse_mean_iou',
    objective_metric_series='validation: epoch_softmax_out_sparse_mean_iou',
    objective_metric_sign='max',

    # optimizer algorithm
    optimizer_class=OptimizerBOHB,
    
    # params
    execution_queue='default', 
    max_number_of_concurrent_tasks=1, 
    optimization_time_limit=1080,  # per task minutes
    compute_time_limit=90,  # total time minutes
    total_max_jobs=100,  
    save_top_k_tasks_only=3,
    min_iteration_per_job=10,
    max_iteration_per_job=200
)
task.execute_remotely(queue_name="services", exit_process=True)

optimizer.set_report_period(1) 
optimizer.start(job_complete_callback=job_complete_callback)  
optimizer.wait()
k=3
top_exp = optimizer.get_top_experiments(top_k=k)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print('Rank {}: task id={} |result={}'
          .format(n, t.id, t.get_last_scalar_metrics()['epoch_softmax_out_sparse_mean_iou']['validation: epoch_softmax_out_sparse_mean_iou']['last']))
optimizer.stop()
print('Optimization done')
task.close()
