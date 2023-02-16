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

TEMPLATE_TASK_ID = 'cc65ed1ecf0d4198bb484c2ef553e5de'
task = Task.init(project_name='Hyperparameter Optimization with BOHB',
                 task_name='Hyperparameter Search: semantic_segmentation',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

optimizer = HyperParameterOptimizer(
    base_task_id=TEMPLATE_TASK_ID,  # experiment to optimize
    # hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('General/epochs', min_value=10, max_value=100, step_size=2),
        UniformIntegerParameterRange('General/batch_size', min_value=8, max_value=20, step_size=4),
        UniformParameterRange('General/base_lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
        UniformIntegerParameterRange('General/first_decay_epoch', min_value=2, max_value=15, step_size=1),
        UniformParameterRange('General/base_lr', min_value=0.1, max_value=0.9, step_size=0.05),
        UniformParameterRange('General/alpha', min_value=1, max_value=3, step_size=0.5),
    ],
    # objective metric
    objective_metric_title='epoch_sparse_mean_iou',
    objective_metric_series='validation: epoch_sparse_mean_iou',
    objective_metric_sign='max',

    # optimizer algorithm
    optimizer_class=OptimizerBOHB,
    
    # params
    execution_queue='default', 
    max_number_of_concurrent_tasks=2, 
    optimization_time_limit=90,  # per task minutes
    compute_time_limit=720,  # total time minutes
    total_max_jobs=50,  
    save_top_k_tasks_only=5,
    min_iteration_per_job=10,
    max_iteration_per_job=100
)
task.execute_remotely(queue_name="services", exit_process=True)

optimizer.set_report_period(1) 
optimizer.start(job_complete_callback=job_complete_callback)  
optimizer.wait()
top_exp = optimizer.get_top_experiments(top_k=3)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print('Rank {}: task id={} |result={}'
          .format(n, t.id, t.get_last_scalar_metrics()['accuracy']['total']['last']))
optimizer.stop()
print('Optimization done')
task.close()
