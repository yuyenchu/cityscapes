from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, ParameterSet
from clearml.automation import HyperParameterOptimizer
from clearml.automation.hpbandster import OptimizerBOHB
from custom_parameters import NargsParameterSet, UniformRange

from clearml import Task
from custom_bohb import CustomOptimizerBOHB

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
configs = {'template_task_id': '6f79b4e40226446ca646f8b91f0e1e50'}
configs = task.connect(configs)

optimizer = HyperParameterOptimizer(
    base_task_id=configs['template_task_id'],  # experiment to optimize
    # hyper-parameters to optimize
    hyper_parameters=[
        # training sizes
        UniformIntegerParameterRange('Args/epochs', min_value=10, max_value=100, step_size=2),
        UniformIntegerParameterRange('Args/kfold', min_value=8, max_value=16, step_size=1),
        UniformIntegerParameterRange('Args/batch_size', min_value=12, max_value=20, step_size=4),
        # learning rate
        UniformParameterRange('Args/initial_lr', min_value=0.0015, max_value=0.005, step_size=0.00025),
        UniformIntegerParameterRange('Args/first_decay_epoch', min_value=2, max_value=17, step_size=1),
        UniformParameterRange('Args/m_mul', min_value=0.5, max_value=0.95, step_size=0.05),
        UniformParameterRange('Args/t_mul', min_value=1, max_value=4, step_size=0.25),
        UniformParameterRange('Args/alpha', min_value=1, max_value=2.75, step_size=0.25),
        # augmentation level
        UniformParameterRange('Args/delta', min_value=0, max_value=0.2, step_size=0.005),
        # loss param and loss weights
        # UniformParameterRange('Args/smooth', min_value=0.0, max_value=0.5, step_size=0.05, include_max_value=False),
        # UniformParameterRange('Args/lambda_val', min_value=0, max_value=0.3, step_size=0.0005),
        NargsParameterSet('Args/smooth', parameter_combinations=[
            UniformRange(min_value=0.0, max_value=0.5, step_size=0.05, include_max_value=False),
            UniformRange(min_value=0.0, max_value=0.5, step_size=0.05, include_max_value=False),
            UniformRange(min_value=0.0, max_value=0.5, step_size=0.05, include_max_value=False),
            UniformRange(min_value=0.0, max_value=0.5, step_size=0.05, include_max_value=False),
        ]),
        NargsParameterSet('Args/lambda_val', parameter_combinations=[
            UniformRange(min_value=0.0, max_value=0.3, step_size=0.001),
            UniformRange(min_value=0.0, max_value=0.3, step_size=0.001),
            UniformRange(min_value=0.0, max_value=0.3, step_size=0.001),
            UniformRange(min_value=0.0, max_value=0.3, step_size=0.001),
        ]),
    ],
    # objective metric
    objective_metric_title='epoch_softmax_out_sparse_mean_iou',
    objective_metric_series='validation: epoch_softmax_out_sparse_mean_iou',
    objective_metric_sign='max_global',

    # optimizer algorithm
    optimizer_class=CustomOptimizerBOHB,
    
    # params
    execution_queue='default', 
    max_number_of_concurrent_tasks=1, 
    optimization_time_limit=2400, # total time minutes
    compute_time_limit=90, # optimize compute time
    total_max_jobs=200,  
    save_top_k_tasks_only=3,
    min_iteration_per_job=20,
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
    t.add_tags('Rank {}'.format(n))
optimizer.stop()
print('Optimization done')
task.close()
