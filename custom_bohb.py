import clearml.automation.hpbandster.bandster as bandster

from nanoid import generate
from time import time
from clearml.automation import DiscreteParameterRange, UniformParameterRange, RandomSeed, UniformIntegerParameterRange, Parameter
from clearml import Task
try:
    # noinspection PyPackageRequirements
    from hpbandster.core.worker import Worker
    # noinspection PyPackageRequirements
    from hpbandster.optimizers import BOHB
    # noinspection PyPackageRequirements
    import hpbandster.core.nameserver as hpns
    # noinspection PyPackageRequirements, PyPep8Naming
    import ConfigSpace as CS
    # noinspection PyPackageRequirements, PyPep8Naming
    import ConfigSpace.hyperparameters as CSH

    Task.add_requirements('hpbandster')
except ImportError:
    raise ImportError("OptimizerBOHB requires 'hpbandster' package, it was not found\n"
                      "install with: pip install hpbandster")

from custom_parameters import NargsParameterSet, UniformRange, UniformIntegerRange

class _CustomTrainsBandsterWorker(bandster.TrainsBandsterWorker):
    def __init__(
            self,
            *args,  # type: Any
            nargs_mapping, 
            **kwargs  # type: Any
    ):
        # type: (...) -> _TrainsBandsterWorker
        super(_CustomTrainsBandsterWorker, self).__init__(*args, **kwargs)
        self.nargs_mapping = nargs_mapping

    def compute(self, config, budget, **kwargs):
        # do somrthing with config
        nargs_config = {k:[config[vv] for vv in v] for k,v in self.nargs_mapping.items()}
        super(_CustomTrainsBandsterWorker, self).compute(config, budget, **kwargs)

class CustomOptimizerBOHB(bandster.OptimizerBOHB):
    def __init__(self, **kwargs):
        super(CustomOptimizerBOHB, self).__init__(**kwargs)
        self.nargs_mapping = {}
    def start(self):
        # type: () -> None
        fake_run_id = 'OptimizerBOHB_{}'.format(time())
        self._namespace = hpns.NameServer(run_id=fake_run_id, host='127.0.0.1', port=self._nameserver_port)
        self._namespace.start()
        budget_iteration_scale = self._max_iteration_per_job
        workers = []
        for i in range(self._num_concurrent_workers):
            w = _CustomTrainsBandsterWorker(
                nargs_mapping=self.nargs_mapping,
                optimizer=self,
                sleep_interval=int(self.pool_period_minutes * 60),
                budget_iteration_scale=budget_iteration_scale,
                base_task_id=self._base_task_id,
                objective=self._objective_metric,
                queue_name=self._execution_queue,
                nameserver='127.0.0.1', nameserver_port=self._nameserver_port, run_id=fake_run_id, id=i)
            w.run(background=True)
            workers.append(w)
        self._bohb = BOHB(configspace=self._convert_hyper_parameters_to_cs(),
                          run_id=fake_run_id,
                          min_budget=float(self._min_iteration_per_job) / float(self._max_iteration_per_job),
                          **self._bohb_kwargs)
        if self.budget.jobs.limit:
            self.budget.jobs.limit *= len(self._bohb.budgets)
        if self.budget.iterations.limit:
            self.budget.iterations.limit *= len(self._bohb.budgets)
        self._res = self._bohb.run(n_iterations=self.total_max_jobs, min_n_workers=self._num_concurrent_workers)
        self.stop()

    def _convert_hyper_parameters_to_cs(self):
        # type: () -> CS.ConfigurationSpace
        cs = CS.ConfigurationSpace(seed=self._seed)
        for p in self._hyper_parameters:
            if isinstance(p, UniformParameterRange):
                hp = CSH.UniformFloatHyperparameter(
                    p.name, lower=p.min_value, upper=p.max_value, log=False, q=p.step_size)
            elif isinstance(p, UniformIntegerParameterRange):
                hp = CSH.UniformIntegerHyperparameter(
                    p.name, lower=p.min_value, upper=p.max_value, log=False, q=p.step_size)
            elif isinstance(p, DiscreteParameterRange):
                hp = CSH.CategoricalHyperparameter(p.name, choices=p.values)
            elif isinstance(p, NargsParameterSet):
                list_name = p.name
                self.nargs_mapping[list_name] = []
                for pp in p:
                    self.nargs_mapping[list_name].append(generate())
                    if isinstance(pp, UniformRange):
                        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                            self.nargs_mapping[list_name][-1], lower=pp.min_value, upper=pp.max_value, log=False, q=pp.step_size))
                    elif isinstance(pp, UniformIntegerRange):
                        cs.add(CSH.UniformIntegerHyperparameter(
                            self.nargs_mapping[list_name][-1], lower=pp.min_value, upper=pp.max_value, log=False, q=pp.step_size))
                    else:
                        raise ValueError("HyperParameter type {} not supported yet inside NargsParameterSet with OptimizerBOHB".format(type(pp)))
            else:
                raise ValueError("HyperParameter type {} not supported yet with OptimizerBOHB".format(type(p)))
            cs.add_hyperparameter(hp)

        return cs